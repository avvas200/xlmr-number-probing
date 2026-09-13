"""
Extract XLM-R hidden states for the target noun in every row of the
balanced CSVs.

For each row:
  1. Feed the pre-tokenized sentence to XLM-R with is_split_into_words=True.
  2. Get hidden states for all 13 layers (embeddings + 12 transformer).
  3. Mean-pool the subword hidden states of the target word (row.word_idx).
  4. Store as float16 to halve disk/memory footprint.

Output: one .pt file per (language, split), each a dict:
    hidden_states : float16 tensor (n_kept, 13, 768)
    labels        : int64 tensor (n_kept,)  0=Sing, 1=Plur
    forms         : list[str] of noun surface forms
    languages     : list[str]
    sent_ids      : list[str]
    layer_index_0_is_embeddings : True   (documentation)

Nouns whose subwords get truncated past --max-length are dropped and
counted. Cast hidden states back to float32 before training probes.

Usage:
    python src/extract_hidden_states.py \\
        --input-dir data/balanced \\
        --output-dir results/hidden_states \\
        --model xlm-roberta-base \\
        --batch-size 32
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


LABEL_MAP = {"Sing": 0, "Plur": 1}


def extract_for_file(
    csv_path: Path,
    output_path: Path,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> None:
    df = pd.read_csv(csv_path)
    n = len(df)

    hidden_list: list[torch.Tensor] = []
    labels: list[int] = []
    forms: list[str] = []
    langs: list[str] = []
    sent_ids: list[str] = []
    dropped = 0

    for start in tqdm(range(0, n, batch_size), desc=csv_path.stem, leave=False):
        batch = df.iloc[start:start + batch_size]
        tokens_batch = [row.tokens.split() for row in batch.itertuples()]
        word_idxs = batch["word_idx"].tolist()

        enc = tokenizer(
            tokens_batch,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states: tuple of 13 tensors of shape (B, S, D)
        # Stack to (B, 13, S, D)
        all_layers = torch.stack(out.hidden_states, dim=1)

        for i, (row, wi) in enumerate(zip(batch.itertuples(), word_idxs)):
            word_ids = enc.word_ids(batch_index=i)
            subword_positions = [j for j, w in enumerate(word_ids) if w == wi]
            if not subword_positions:
                # Target word was truncated away.
                dropped += 1
                continue
            # all_layers[i] shape: (13, S, D). Advanced-index the S dim,
            # then mean-pool subwords to get (13, D).
            pooled = all_layers[i][:, subword_positions, :].mean(dim=1)
            hidden_list.append(pooled.half().cpu())
            labels.append(LABEL_MAP[row.number])
            forms.append(row.form)
            langs.append(row.language)
            sent_ids.append(row.sent_id)

    # Free GPU memory before the next file.
    del all_layers, out, enc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not hidden_list:
        raise RuntimeError(f"No usable rows in {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_states": torch.stack(hidden_list),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "forms": forms,
            "languages": langs,
            "sent_ids": sent_ids,
            "layer_index_0_is_embeddings": True,
        },
        output_path,
    )
    print(
        f"  {csv_path.stem}: kept={len(hidden_list)}, dropped(trunc)={dropped}, "
        f"→ {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/balanced"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hidden_states"))
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    csvs = sorted(args.input_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs in {args.input_dir}")

    for csv_path in csvs:
        output_path = args.output_dir / f"{csv_path.stem}.pt"
        if output_path.exists():
            print(f"  {output_path.name} already exists — skipping")
            continue
        extract_for_file(
            csv_path, output_path, tokenizer, model,
            device, args.batch_size, args.max_length,
        )
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
"""
Extract noun tokens with their grammatical number labels from a
Universal Dependencies (UD) treebank in CoNLL-U format.

For each noun (UPOS = NOUN) whose Number feature is 'Sing' or 'Plur',
we record:
    - the full sentence (for later hidden-state extraction)
    - the token's 1-indexed position in the sentence
    - the surface form and lemma
    - the number label

Nouns without a Number feature, or with values other than Sing/Plur
(e.g. Arabic's 'Dual'), are skipped so the probing task is a clean
binary classification. Dual could be added later as a follow-up.

Usage:
    python src/extract_ud.py \\
        --input data/raw/UD_English-EWT/en_ewt-ud-train.conllu \\
        --output data/processed/english.csv \\
        --language en
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from conllu import parse_incr


FIELDS = [
    "language", "sent_id", "sentence", "tokens", "word_idx",
    "token_id", "form", "lemma", "number",
]


def extract_nouns(conllu_path: Path, language: str) -> list[dict]:
    """Return one row per singular/plural noun in the treebank."""
    rows: list[dict] = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            sent_id = sentence.metadata.get("sent_id", "")
            sent_text = sentence.metadata.get("text", "")

            # We need the raw sentence text to feed the model later.
            if not sent_text:
                continue

            # Build the underlying-token list once per sentence. We skip
            # multi-word tokens (tuple ids) and empty nodes (decimal ids).
            # This list is what we later feed to XLM-R with
            # `is_split_into_words=True`, so subword alignment is unambiguous.
            integer_tokens = [t for t in sentence if isinstance(t["id"], int)]
            tokens_str = " ".join(t["form"] for t in integer_tokens)

            for word_idx, token in enumerate(integer_tokens):
                if token["upos"] != "NOUN":
                    continue

                feats = token.get("feats") or {}
                number = feats.get("Number")

                # Keep only clear singular/plural cases.
                if number not in {"Sing", "Plur"}:
                    continue

                rows.append({
                    "language": language,
                    "sent_id": sent_id,
                    "sentence": sent_text,
                    "tokens": tokens_str,
                    "word_idx": word_idx,          # 0-indexed into `tokens`
                    "token_id": token["id"],       # original CoNLL-U id (1-indexed)
                    "form": token["form"],
                    "lemma": token["lemma"] or "",
                    "number": number,
                })
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict], language: str, output_path: Path) -> None:
    counts = Counter(r["number"] for r in rows)
    sing = counts.get("Sing", 0)
    plur = counts.get("Plur", 0)
    total = sing + plur
    balance = (min(sing, plur) / max(sing, plur)) if max(sing, plur) else 0.0

    print(f"Language: {language}")
    print(f"  Total nouns extracted: {total}")
    print(f"  Singular: {sing}")
    print(f"  Plural:   {plur}")
    print(f"  Class balance (min/max): {balance:.2f}")
    print(f"  Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to .conllu file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV")
    parser.add_argument("--language", type=str, required=True, help="Language code, e.g. 'en'")
    args = parser.parse_args()

    rows = extract_nouns(args.input, args.language)
    write_csv(rows, args.output)
    print_stats(rows, args.language, args.output)


if __name__ == "__main__":
    main()
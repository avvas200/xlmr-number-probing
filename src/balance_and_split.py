"""
Balance and split extracted noun data for cross-lingual probing.

For each language CSV in --input-dir:
  1. Split *sentences* (not nouns) into train / val / test to prevent
     leakage from nouns in the same sentence appearing in two splits.
  2. Within each split, undersample the majority class so singular and
     plural are exactly balanced. A probe that always predicts the
     majority class then gets 50% accuracy, so any lift is real.
  3. Cap each language at --target-per-class per class in total, so
     cross-lingual comparisons are fair.

Usage:
    python src/balance_and_split.py \\
        --input-dir data/processed \\
        --output-dir data/balanced \\
        --target-per-class 3000 \\
        --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}


def balance_and_split(
    df: pd.DataFrame,
    target_per_class: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)

    # Split sentence ids, not rows.
    sent_ids = np.array(df["sent_id"].unique(), dtype=object)
    rng.shuffle(sent_ids)

    n = len(sent_ids)
    train_end = int(n * SPLIT_FRACTIONS["train"])
    val_end = int(n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]))

    id_splits = {
        "train": set(sent_ids[:train_end]),
        "val": set(sent_ids[train_end:val_end]),
        "test": set(sent_ids[val_end:]),
    }

    splits: dict[str, pd.DataFrame] = {}
    for name, ids in id_splits.items():
        split_df = df[df["sent_id"].isin(ids)]
        sing = split_df[split_df["number"] == "Sing"]
        plur = split_df[split_df["number"] == "Plur"]

        target = int(target_per_class * SPLIT_FRACTIONS[name])
        n_take = min(target, len(sing), len(plur))

        sing = sing.sample(n=n_take, random_state=seed)
        plur = plur.sample(n=n_take, random_state=seed)

        balanced = (
            pd.concat([sing, plur])
            .sample(frac=1, random_state=seed)  # shuffle
            .reset_index(drop=True)
        )
        splits[name] = balanced

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/balanced"))
    parser.add_argument("--target-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(args.input_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found in {args.input_dir}")

    print(f"Processing {len(csvs)} languages")
    print(f"Target per class: {args.target_per_class}")
    print(f"Splits: {SPLIT_FRACTIONS}")
    print(f"Seed: {args.seed}\n")

    for csv_path in csvs:
        language = csv_path.stem
        df = pd.read_csv(csv_path)

        splits = balance_and_split(df, args.target_per_class, args.seed)

        print(f"{language}:")
        for name, split_df in splits.items():
            out = args.output_dir / f"{language}_{name}.csv"
            split_df.to_csv(out, index=False)
            counts = split_df["number"].value_counts()
            print(
                f"  {name:5s}: {len(split_df):5d} rows "
                f"(Sing={counts.get('Sing', 0)}, Plur={counts.get('Plur', 0)})"
            )
        print()


if __name__ == "__main__":
    main()
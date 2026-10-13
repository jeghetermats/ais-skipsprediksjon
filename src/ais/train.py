"""Trener GRU-modellen på april–juli og velger beste epoke på august.
Kjør: ais-train --trips data/trips.parquet"""

import argparse
from pathlib import Path

import pandas as pd

from ais.evaluate import split
from ais.model import save, train
from ais.windows import make_windows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trips", type=Path, default=Path("data/trips.parquet"))
    p.add_argument("--out", type=Path, default=Path("models/gru.pt"))
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    trips = pd.read_parquet(args.trips)
    train_w = split(make_windows(trips, stride=1))["train"]  # flere, overlappende eksempler
    val_w = split(make_windows(trips))["val"]
    print(f"{len(train_w)} treningseksempler, {len(val_w)} valideringseksempler")
    save(train(train_w, val_w, epochs=args.epochs), args.out)
    print(f"Lagret {args.out}")


if __name__ == "__main__":
    main()

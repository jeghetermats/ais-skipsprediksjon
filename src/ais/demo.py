"""Lager små filer med spor og prediksjoner for et utvalg september-turer til Streamlit-demoen.
Velger turene i fart med flest svinger (der modellene skiller seg mest) og noen tilfeldige.
Kjør: ais-demo-data"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ais.baselines import RouteBaseline, constant_velocity
from ais.evaluate import TEST_MONTHS, split, unseen_ship
from ais.prepare import to_latlon
from ais.windows import HORIZONS, make_windows

N_TURNS, N_RANDOM = 20, 10
MIN_MEDIAN_SOG = 5.0  # knop; sakte turer er lite interessante å se på


def pick_trips(situation: pd.DataFrame, seed: int = 0) -> list[int]:
    """Turene med flest svinger, pluss noen tilfeldige blant resten."""
    turns = situation.groupby("trip_id")["situation"].apply(lambda s: (s == "sving").sum())
    top = turns.nlargest(N_TURNS).index.tolist()
    rest = turns.index.difference(top)
    rnd = np.random.default_rng(seed).choice(rest, min(N_RANDOM, len(rest)), replace=False)
    return top + rnd.tolist()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trips", type=Path, default=Path("data/trips.parquet"))
    p.add_argument("--model", type=Path, default=Path("models/gru.pt"))
    p.add_argument("--out", type=Path, default=Path("app/data"))
    args = p.parse_args()

    trips = pd.read_parquet(args.trips)
    route = RouteBaseline().fit(split(make_windows(trips))["train"])
    sept = trips[trips["time"].dt.month.isin(TEST_MONTHS)]
    w = make_windows(sept, stride=1)  # ett eksempel per minutt, så glidebryteren går jevnt
    fast = sept.groupby("trip_id")["sog"].median().loc[lambda s: s >= MIN_MEDIAN_SOG].index
    w = w.subset(w.meta["trip_id"].isin(fast).to_numpy())
    chosen = pick_trips(pd.DataFrame({"trip_id": w.meta["trip_id"], "situation": w.situation}))
    w = w.subset(w.meta["trip_id"].isin(chosen).to_numpy())

    preds = {"Fysikk": constant_velocity(w), "Rute": route.predict(w)}
    if args.model.exists():
        from ais.model import load, predict  # torch lastes bare når modellen finnes
        preds["GRU"] = predict(load(args.model), w)

    rows = []
    for name, pred in preds.items():
        err = np.linalg.norm(pred - w.target, axis=2)
        for j, h in enumerate(HORIZONS):
            lat, lon = to_latlon(w.pos[:, 0] + pred[:, j, 0], w.pos[:, 1] + pred[:, j, 1])
            rows.append(w.meta.assign(model=name, horizon=h, lat=lat, lon=lon, error_m=err[:, j],
                                      situation=w.situation))
    predictions = pd.concat(rows, ignore_index=True)
    predictions["new_ship"] = unseen_ship(predictions["mmsi"])

    args.out.mkdir(parents=True, exist_ok=True)
    sept[sept["trip_id"].isin(chosen)][["mmsi", "trip_id", "time", "lat", "lon", "sog"]].to_parquet(
        args.out / "tracks.parquet", index=False)
    predictions.to_parquet(args.out / "predictions.parquet", index=False)
    print(f"{len(chosen)} turer, modeller: {', '.join(preds)} -> {args.out}")


if __name__ == "__main__":
    main()

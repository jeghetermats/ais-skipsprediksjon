"""Evaluerer baselines (og GRU-modellen hvis den er trent) på september, delt på situasjon og på
skip modellen har sett eller ikke. Kjør: ais-evaluate --trips data/trips.parquet"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ais.baselines import RouteBaseline, constant_velocity
from ais.windows import HORIZONS, Windows, make_windows

TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS = (4, 5, 6, 7), (8,), (9,)


def unseen_ship(mmsi) -> np.ndarray:
    """Omtrent hvert femte skip holdes helt utenfor trening og validering.

    Bruker en hash, fordi MMSI-nummer ofte ender på 0 og `mmsi % 5` da velger nesten alle skip.
    """
    mmsi = pd.Series(np.asarray(mmsi))
    codes = {m: zlib.crc32(str(m).encode()) % 5 == 0 for m in mmsi.unique()}
    return mmsi.map(codes).to_numpy(bool)


def split(w: Windows) -> dict[str, Windows]:
    month = w.meta["time"].dt.month.to_numpy()
    unseen = unseen_ship(w.meta["mmsi"])
    sets = {
        "train": np.isin(month, TRAIN_MONTHS) & ~unseen,
        "val": np.isin(month, VAL_MONTHS) & ~unseen,
        "test": np.isin(month, TEST_MONTHS),
    }
    return {k: w.subset(m) for k, m in sets.items()}


def errors(pred: np.ndarray, w: Windows) -> np.ndarray:
    """Avstand i meter mellom forutsagt og faktisk posisjon, (n, len(HORIZONS))."""
    return np.linalg.norm(pred - w.target, axis=2)


def mean_ci(err: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Snitt med 95 % konfidensintervall, der hele turer trekkes om igjen (bootstrap)."""
    codes, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, err, len(codes)), np.bincount(inv, minlength=len(codes))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return err.mean(), *np.percentile(boot, [2.5, 97.5])


def report(preds: dict[str, np.ndarray], w: Windows) -> pd.DataFrame:
    rows = []
    groups = {"alle": np.ones(len(w), bool),
              "skip sett i trening": ~unseen_ship(w.meta["mmsi"]),
              "nye skip": unseen_ship(w.meta["mmsi"])}
    situations = ("rett kurs", "sving", "fartsendring", "lav fart")
    groups |= {f"situasjon: {s}": w.situation == s for s in situations}
    trips = w.meta["trip_id"].to_numpy()
    for gname, mask in groups.items():
        for model, pred in preds.items():
            e = errors(pred[mask], w.subset(mask))
            for j, h in enumerate(HORIZONS):
                m, lo, hi = mean_ci(e[:, j], trips[mask])
                rows.append({"gruppe": gname, "modell": model, "n": int(mask.sum()), "horisont": h,
                             "snitt": round(float(m), 1), "lav": round(float(lo), 1),
                             "hoy": round(float(hi), 1)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trips", type=Path, default=Path("data/trips.parquet"))
    p.add_argument("--model", type=Path, default=Path("models/gru.pt"))
    p.add_argument("--out", type=Path, default=Path("results/results.md"))
    args = p.parse_args()
    sets = split(make_windows(pd.read_parquet(args.trips)))
    test = sets["test"]
    preds = {"fysikk": constant_velocity(test),
             "rute": RouteBaseline().fit(sets["train"]).predict(test)}
    if args.model.exists():
        from ais.model import load, predict  # torch lastes bare når modellen finnes
        preds["GRU"] = predict(load(args.model), test)
    res = report(preds, test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_json(args.out.with_suffix(".json"), orient="records", indent=1)  # leses av appen
    res["cell"] = res.apply(lambda r: f"{r.snitt:.0f} ({r.lav:.0f}–{r.hoy:.0f})", axis=1)
    table = res.pivot(index=["gruppe", "modell", "n"], columns="horisont", values="cell")
    table = table.loc[res[["gruppe", "modell", "n"]].drop_duplicates().itertuples(index=False)]
    table.columns = [f"{h} min" for h in table.columns]
    table = table.reset_index()
    rows = [table.columns, ["---"] * len(table.columns), *table.itertuples(index=False)]
    md = ("Gjennomsnittlig feil i meter på testsettet (september), med 95 % konfidensintervall.\n\n"
          + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows))
    args.out.write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

"""Gjør Kystverkets historiske AIS-filer (HAIS, Parquet) om til turer med faste 1-minutters steg.

En tur er en sammenhengende periode der skipet er i fart (minst MIN_SOG knop) uten hull lengre
enn MAX_GAP. Kjør: ais-prepare --raw data/raw --out data/trips.parquet
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MIN_SOG = 0.5  # knop; lavere regnes som at skipet ligger stille
MAX_GAP = pd.Timedelta("5min")  # lengre hull deler en tur i to
MAX_JUMP_KN = 50.0  # posisjonshopp som tilsvarer høyere fart enn dette, er feil
MIN_TRIP_MIN = 30  # kortere turer gir for lite historikk til å trene på
STEP = pd.Timedelta("1min")

# shortcut: lokal flat projeksjon rundt fjorden (feil < 0,5 % innenfor området), bytt til UTM ved
# større områder
LAT0, LON0 = 63.6, 10.5
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(LAT0))


def to_xy(lat, lon):
    """Breddegrad/lengdegrad til meter øst (x) og nord (y) for et fast punkt midt i fjorden."""
    return (np.asarray(lon) - LON0) * M_PER_DEG_LON, (np.asarray(lat) - LAT0) * M_PER_DEG_LAT


def to_latlon(x, y):
    """Motsatt av to_xy."""
    return np.asarray(y) / M_PER_DEG_LAT + LAT0, np.asarray(x) / M_PER_DEG_LON + LON0


def load_positions(raw: Path) -> pd.DataFrame:
    """Leser alle Parquet-filer under raw og fjerner ugyldige og dupliserte meldinger."""
    files = str(raw / "**" / "*.parquet")
    return duckdb.sql(f"""
        SELECT DISTINCT ON (mmsi, date_time_utc)
            mmsi, date_time_utc AS time, latitude AS lat, longitude AS lon,
            CAST(speed_over_ground AS DOUBLE) AS sog
        FROM read_parquet('{files}')
        WHERE latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180
          AND speed_over_ground < 102.3  -- 102.3 betyr "ikke tilgjengelig"
        ORDER BY mmsi, time
    """).df()


def drop_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """Fjerner punkter som ligger urealistisk langt fra forrige punkt for samme skip."""
    x, y = to_xy(df["lat"], df["lon"])
    same = df["mmsi"].eq(df["mmsi"].shift())
    dist = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    dt_s = df["time"].diff().dt.total_seconds()
    speed_kn = dist / dt_s.clip(lower=1) / 0.514444
    return df[~(same & (speed_kn > MAX_JUMP_KN))].reset_index(drop=True)


def resample_trip(g: pd.DataFrame) -> pd.DataFrame:
    """Interpolerer én tur til faste 1-minutters steg."""
    t = g["time"].astype("int64").to_numpy() / 1e9
    grid = pd.date_range(g["time"].iloc[0].ceil(STEP), g["time"].iloc[-1].floor(STEP), freq=STEP)
    tg = grid.astype("int64").to_numpy() / 1e9
    return pd.DataFrame({
        "time": grid,
        "lat": np.interp(tg, t, g["lat"]),
        "lon": np.interp(tg, t, g["lon"]),
        "sog": np.interp(tg, t, g["sog"]),
    })


def make_trips(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_jumps(df[df["sog"] >= MIN_SOG].reset_index(drop=True))
    new_trip = df["mmsi"].ne(df["mmsi"].shift()) | df["time"].diff().gt(MAX_GAP)
    df["trip_id"] = new_trip.cumsum() - 1
    parts = []
    for (mmsi, trip_id), g in df.groupby(["mmsi", "trip_id"], sort=False):
        r = resample_trip(g)
        if len(r) >= MIN_TRIP_MIN:
            parts.append(r.assign(mmsi=mmsi, trip_id=trip_id))
    cols = ["mmsi", "trip_id", "time", "lat", "lon", "x", "y", "sog"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["trip_id"] = out.groupby(["mmsi", "trip_id"], sort=False).ngroup()  # 0, 1, 2 ...
    out["x"], out["y"] = to_xy(out["lat"], out["lon"])
    return out[cols]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/trips.parquet"))
    args = p.parse_args()
    trips = make_trips(load_positions(args.raw))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(args.out, index=False)
    print(f"{trips['trip_id'].nunique()} turer, {trips['mmsi'].nunique()} skip, "
          f"{len(trips)} minutter -> {args.out}")


if __name__ == "__main__":
    main()

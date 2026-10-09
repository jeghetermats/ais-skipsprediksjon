import numpy as np
import pandas as pd

from ais.prepare import make_trips, to_xy

T0 = pd.Timestamp("2026-10-14 12:00:00")


def track(mmsi, start, minutes, sog=10.0, lat0=63.5, dlat_per_min=0.001):
    """Et skip som går rett nordover med en melding hvert 6. sekund."""
    t = pd.date_range(start, periods=minutes * 10, freq="6s")
    lat = lat0 + dlat_per_min * np.arange(len(t)) / 10
    return pd.DataFrame({"mmsi": mmsi, "time": t, "lat": lat, "lon": 10.4, "sog": sog})


def test_resamples_to_one_minute_and_interpolates_position():
    out = make_trips(track(1, T0, 60))
    assert out["time"].diff().dropna().eq(pd.Timedelta("1min")).all()
    row = out[out["time"] == T0 + pd.Timedelta("10min")].iloc[0]
    assert abs(row["lat"] - 63.51) < 1e-9


def test_gap_splits_trip_and_short_trips_are_dropped():
    a = track(1, T0, 40)
    b = track(1, T0 + pd.Timedelta("50min"), 40, lat0=63.6)  # 10 min hull
    c = track(1, T0 + pd.Timedelta("100min"), 10, lat0=63.7)  # for kort
    out = make_trips(pd.concat([a, b, c], ignore_index=True))
    assert out["trip_id"].nunique() == 2


def test_stationary_points_and_position_jumps_are_removed():
    df = track(1, T0, 40)
    df.loc[100, "lat"] += 1.0  # 111 km hopp på 6 s
    still = track(1, T0 + pd.Timedelta("40min"), 40, sog=0.0)
    out = make_trips(pd.concat([df, still], ignore_index=True))
    assert out["lat"].max() < 63.6 and out["time"].max() < T0 + pd.Timedelta("40min")


def test_ships_are_kept_apart():
    out = make_trips(pd.concat([track(1, T0, 40), track(2, T0, 40)], ignore_index=True))
    assert out.groupby("trip_id")["mmsi"].nunique().eq(1).all() and out["trip_id"].nunique() == 2


def test_to_xy_one_km_north():
    x, y = to_xy([63.6 + 1000 / 111_320], [10.5])
    assert abs(x[0]) < 1e-6 and abs(y[0] - 1000) < 1e-6


def test_to_latlon_inverts_to_xy():
    from ais.prepare import to_latlon

    lat, lon = to_latlon(*to_xy([63.45, 63.9], [10.4, 11.2]))
    assert np.allclose(lat, [63.45, 63.9]) and np.allclose(lon, [10.4, 11.2])

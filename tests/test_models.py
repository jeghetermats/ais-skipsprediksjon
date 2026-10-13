import numpy as np
import pandas as pd
import pytest

from ais.baselines import RouteBaseline, constant_velocity
from ais.evaluate import mean_ci, unseen_ship
from ais.windows import HIST, HORIZONS, make_windows

T0 = pd.Timestamp("2026-06-01 12:00")


def trip(trip_id, minutes=100, turn_at=None, mmsi=1):
    """1-minutters tur rett østover i 300 m/min, eventuelt med en 90-graders sving nordover."""
    x, y = 300.0 * np.arange(minutes), np.zeros(minutes)
    if turn_at is not None:
        k = np.arange(minutes) - turn_at
        x = np.where(k > 0, 300.0 * turn_at, x)
        y = np.where(k > 0, 300.0 * k, 0.0)
    return pd.DataFrame({"mmsi": mmsi, "trip_id": trip_id,
                         "time": pd.date_range(T0, periods=minutes, freq="1min"),
                         "x": x, "y": y, "sog": 9.7})


def test_windows_have_right_shape_and_target():
    w = make_windows(trip(0), stride=5)
    assert w.hist.shape[1:] == (HIST, 2) and w.target.shape[1:] == (len(HORIZONS), 2)
    assert np.allclose(w.hist[:, -1], 0)
    assert np.allclose(w.target[:, :, 0], [300.0 * h for h in HORIZONS])  # rett østover
    assert set(w.situation) == {"rett kurs"}


def test_turn_is_labelled_and_physics_is_exact_only_on_straight_line():
    straight, turning = make_windows(trip(0)), make_windows(trip(1, turn_at=50))
    assert np.allclose(constant_velocity(straight), straight.target)
    assert "sving" in set(turning.situation)
    assert not np.allclose(constant_velocity(turning), turning.target)


def test_route_baseline_learns_the_turn():
    train_w = make_windows(trip(0, turn_at=50), stride=1)
    test_w = make_windows(trip(1, turn_at=50), stride=1)
    err = np.linalg.norm(RouteBaseline(k=1).fit(train_w).predict(test_w) - test_w.target, axis=2)
    assert err.max() < 1e-6


def test_unseen_ships_are_not_just_mmsi_ending_in_zero():
    mmsi = np.arange(257_000_000, 257_100_000, 10)  # alle ender på 0
    share = unseen_ship(mmsi).mean()
    assert 0.15 < share < 0.25 and (unseen_ship(mmsi) == unseen_ship(mmsi)).all()


def test_mean_ci_contains_mean():
    err, groups = np.random.default_rng(0).random(500), np.repeat(np.arange(50), 10)
    m, lo, hi = mean_ci(err, groups)
    assert lo < m < hi


def test_gru_trains_and_predicts():
    pytest.importorskip("torch")
    from ais.model import predict, train

    w = make_windows(pd.concat([trip(i, turn_at=50) for i in range(4)]), stride=1)
    model = train(w, w, epochs=1, log=lambda _: None)
    assert predict(model, w).shape == w.target.shape


def test_slow_ship_is_not_labelled_as_turn():
    t = trip(0, turn_at=50)
    t[["x", "y"]] *= 0.1  # 30 m/min, ca. 1 knop
    assert "sving" not in set(make_windows(t).situation)

"""Lager treningseksempler: de siste HIST minuttene av en tur og hvor skipet er HORIZONS minutter
senere. Alle posisjoner er relative til skipets posisjon i ankerøyeblikket (meter)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

HIST = 30
HORIZONS = (10, 20, 30)
LOW_SPEED = 2 * 1852 / 60  # 2 knop i meter per minutt


@dataclass
class Windows:
    hist: np.ndarray  # (n, HIST, 2) tidligere posisjoner relativt til nåværende, siste rad er 0
    pos: np.ndarray  # (n, 2) nåværende posisjon i fjordens koordinater (x, y)
    sog: np.ndarray  # (n, HIST) fart i knop
    target: np.ndarray  # (n, len(HORIZONS), 2) forflytning ved hver horisont
    meta: pd.DataFrame  # mmsi, trip_id, time for ankerøyeblikket
    situation: np.ndarray  # (n,) "rett kurs", "sving", "fartsendring", "lav fart" eller "annet"

    def subset(self, mask: np.ndarray) -> "Windows":
        return Windows(self.hist[mask], self.pos[mask], self.sog[mask], self.target[mask],
                       self.meta[mask].reset_index(drop=True), self.situation[mask])

    def __len__(self) -> int:
        return len(self.pos)


def situation_labels(hist: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Klassifiserer hva skipet gjør de neste 30 minuttene, ut fra fart og kurs nå og på slutten."""
    v_now = (hist[:, -1] - hist[:, -3]) / 2
    v_end = (target[:, -1] - target[:, -2]) / 10  # gjennomsnitt siste 10 min
    speed_now, speed_end = np.linalg.norm(v_now, axis=1), np.linalg.norm(v_end, axis=1)
    cos = (v_now * v_end).sum(1) / np.maximum(speed_now * speed_end, 1e-9)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ratio = speed_end / np.maximum(speed_now, 1e-9)
    slow = speed_now < LOW_SPEED  # under 2 knop er kursen mest støy
    return np.select(
        [slow, turn >= 30, (ratio < 0.67) | (ratio > 1.5),
         (turn < 10) & (ratio > 0.8) & (ratio < 1.25)],
        ["lav fart", "sving", "fartsendring", "rett kurs"], default="annet")


def make_windows(trips: pd.DataFrame, stride: int = 5) -> Windows:
    """Ett eksempel per `stride` minutter i hver tur som har nok historikk og framtid."""
    max_h = max(HORIZONS)
    hist, pos, sog, target, meta = [], [], [], [], []
    for (mmsi, trip_id), g in trips.groupby(["mmsi", "trip_id"], sort=False):
        n = len(g)
        anchors = np.arange(HIST - 1, n - max_h, stride)
        if len(anchors) == 0:
            continue
        xy = g[["x", "y"]].to_numpy()
        s = g["sog"].to_numpy()
        idx = anchors[:, None] + np.arange(-HIST + 1, 1)  # (a, HIST)
        cur = xy[anchors]
        hist.append(xy[idx] - cur[:, None])
        pos.append(cur)
        sog.append(s[idx])
        target.append(np.stack([xy[anchors + h] - cur for h in HORIZONS], axis=1))
        meta.append(pd.DataFrame({"mmsi": mmsi, "trip_id": trip_id,
                                  "time": g["time"].to_numpy()[anchors]}))
    hist, target = np.concatenate(hist), np.concatenate(target)
    return Windows(hist.astype(np.float32), np.concatenate(pos).astype(np.float32),
                   np.concatenate(sog).astype(np.float32), target.astype(np.float32),
                   pd.concat(meta, ignore_index=True), situation_labels(hist, target))

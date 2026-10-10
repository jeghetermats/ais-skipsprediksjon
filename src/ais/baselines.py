"""To baselines som GRU-modellen må slå."""

import numpy as np
from scipy.spatial import cKDTree

from ais.windows import HORIZONS, Windows


def constant_velocity(w: Windows) -> np.ndarray:
    """Fysikkbaseline: skipet holder farten og kursen fra de to siste minuttene."""
    v = (w.hist[:, -1] - w.hist[:, -3]) / 2  # meter per minutt
    return np.stack([v * h for h in HORIZONS], axis=1)


class RouteBaseline:
    """Rutebaseline: gjør det skip på samme sted, med lik fart og kurs, gjorde i treningsdataene.

    Nabosøket bruker (x, y, VEL_WEIGHT * vx, VEL_WEIGHT * vy), slik at 1 knop forskjell i fart
    (ca. 31 m/min) teller omtrent som 100 m avstand.
    """

    VEL_WEIGHT = 3.0

    def __init__(self, k: int = 25):
        self.k = k

    def _features(self, w: Windows) -> np.ndarray:
        v = (w.hist[:, -1] - w.hist[:, -3]) / 2
        return np.hstack([w.pos, self.VEL_WEIGHT * v])

    def fit(self, w: Windows) -> "RouteBaseline":
        self.tree = cKDTree(self._features(w))
        self.target = w.target
        return self

    def predict(self, w: Windows) -> np.ndarray:
        _, idx = self.tree.query(self._features(w), k=self.k, workers=-1)
        idx = idx.reshape(len(idx), -1)  # med k=1 gir cKDTree en 1D-array
        return np.median(self.target[idx], axis=1)

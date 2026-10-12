"""GRU som forutsier hvor mye skipet avviker fra fysikkbaselinen ved hver horisont.

Ved å lære avviket i stedet for hele forflytningen blir rett kurs gratis, og modellen kan bruke
kapasiteten på svinger og fartsendringer.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ais.baselines import constant_velocity
from ais.windows import HIST, HORIZONS, Windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIST_SCALE, STEP_SCALE, POS_SCALE, SOG_SCALE, OUT_SCALE = 10_000.0, 300.0, 50_000.0, 20.0, 1000.0
N_FEATURES = 7  # posisjon relativt nå (2), steg per minutt (2), fart (1), posisjon i fjorden (2)


def features(w: Windows) -> np.ndarray:
    steps = np.diff(w.hist, axis=1, prepend=w.hist[:, :1])
    pos = np.repeat(w.pos[:, None] / POS_SCALE, HIST, axis=1)
    return np.concatenate([w.hist / HIST_SCALE, steps / STEP_SCALE,
                           w.sog[..., None] / SOG_SCALE, pos], axis=2).astype(np.float32)


class GRUModel(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(N_FEATURES, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, len(HORIZONS) * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1]).view(-1, len(HORIZONS), 2)


def _tensors(w: Windows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cv = constant_velocity(w)
    return (torch.from_numpy(features(w)), torch.from_numpy(cv / OUT_SCALE),
            torch.from_numpy(w.target / OUT_SCALE))


@torch.no_grad()
def predict(model: GRUModel, w: Windows, batch: int = 4096) -> np.ndarray:
    model.eval().to(DEVICE)
    x, cv, _ = _tensors(w)
    out = torch.cat([model(x[i:i + batch].to(DEVICE)).cpu() for i in range(0, len(x), batch)])
    return ((out + cv) * OUT_SCALE).numpy()


def mean_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Samme mål som evalueringen: gjennomsnittlig avstand (i km) over alle horisonter."""
    return torch.linalg.vector_norm(pred - target, dim=2).mean()


def train(train_w: Windows, val_w: Windows, epochs: int = 15, lr: float = 1e-3, batch: int = 512,
          patience: int = 3, seed: int = 0, log=print) -> GRUModel:
    """Trener med tidlig stopp på valideringsfeilen og returnerer den beste modellen."""
    torch.manual_seed(seed)
    model = GRUModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, cv, y = (t.to(DEVICE) for t in _tensors(train_w))
    xv, cvv, yv = (t.to(DEVICE) for t in _tensors(val_w))
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x), device=DEVICE)
        for i in range(0, len(x), batch):
            j = perm[i:i + batch]
            loss = mean_distance(model(x[j]) + cv[j], y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(xv[i:i + 4096]) for i in range(0, len(xv), 4096)]) + cvv
            val = float(mean_distance(pred, yv)) * OUT_SCALE
        log(f"epoke {epoch + 1}: valideringsfeil {val:.0f} m (snitt over horisontene)")
        if val < best:
            best, best_state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def save(model: GRUModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def load(path: Path) -> GRUModel:
    model = GRUModel()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return model

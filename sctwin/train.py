"""Training / evaluation utilities shared by all forecasting domains."""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(max(1, torch.get_num_threads()))


class Scaler:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6; return self
    def tf(self, X): return (X - self.mu) / self.sd
    def inv(self, X, cols=None):
        mu, sd = (self.mu, self.sd) if cols is None else (self.mu[cols], self.sd[cols])
        return X * sd + mu


def split(n, train=0.7, val=0.1):
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)


def fit(model, Xtr, Ytr, Xva, Yva, epochs=30, lr=1e-3, bs=64, patience=6, verbose=True):
    if not any(p.requires_grad and p.numel() > 1 for p in model.parameters()):
        return model, 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), batch_size=bs, shuffle=True)
    Xv, Yv = torch.tensor(Xva), torch.tensor(Yva)
    best, best_state, bad, t0 = 1e9, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = nn.functional.l1_loss(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad(): v = nn.functional.l1_loss(model(Xv), Yv).item()
        sched.step(v)
        if v < best - 1e-4: best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else: bad += 1
        if verbose: print(f"    ep{ep:02d} val_MAE(z)={v:.4f}")
        if bad >= patience: break
    model.load_state_dict(best_state)
    return model, time.time() - t0


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    return np.concatenate([model(torch.tensor(X[i:i + bs])).numpy() for i in range(0, len(X), bs)])


def metrics(y, p, mape_floor=1.0):
    e = p - y
    m = np.abs(y) > mape_floor
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                MAPE=float((np.abs(e[m]) / np.abs(y[m])).mean() * 100))


def horizon_metrics(y, p):
    return [metrics(y[:, h], p[:, h])["MAE"] for h in range(y.shape[1])]

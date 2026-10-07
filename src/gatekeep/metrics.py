"""Small, well-tested measurement helpers."""
import numpy as np
from sklearn.metrics import f1_score


def ece(conf, correct, bins=10):
    """Expected calibration error: how far stated confidence is from actual accuracy."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def coverage_curve(conf, correct, taus):
    """For each tau: (fraction answered with conf >= tau, accuracy on those)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    out = []
    for t in taus:
        m = conf >= t
        out.append((t, float(m.mean()), float(correct[m].mean()) if m.any() else float("nan")))
    return out


def pick_tau(conf, correct, target, taus=None):
    taus = np.round(np.arange(0.5, 1.0, 0.01), 2) if taus is None else taus
    for t, _, acc in coverage_curve(conf, correct, taus):
        if acc == acc and acc >= target:  # acc == acc filters NaN
            return float(t)
    return 1.01


def bootstrap_ci(values, n=1000, seed=0):
    v, rng = np.asarray(values, float), np.random.default_rng(seed)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro"))

"""Pruebas del autoencoder (sintéticas, rápidas).  Ejecutar: pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neural import Autoencoder  # noqa: E402


def test_gradients_match_finite_differences():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 5)
    ae = Autoencoder(5, 8, 3, seed=1)
    _, grads = ae.loss_and_grads(X)
    eps, worst = 1e-6, 0.0
    for k in ae.PARAMS:
        flat = ae.p[k].ravel()
        for i in rng.choice(flat.size, min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; lp = ae.loss(X)
            flat[i] = old - eps; lm = ae.loss(X)
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), grads[k].ravel()[i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    assert worst < 1e-5, worst


def test_training_reduces_loss_and_save_load(tmp_path):
    rng = np.random.RandomState(0)
    base = rng.randn(800, 2)
    X = np.c_[base, base[:, 0] + base[:, 1], base[:, 0] - base[:, 1], 0.5 * base[:, 1]] + 0.05 * rng.randn(800, 5)
    ae = Autoencoder(5, 8, 2, seed=3)
    before = ae.loss(X)
    ae.fit(X, epochs=40, batch_size=64, lr=5e-3, patience=10)
    assert ae.loss(X) < 0.3 * before and ae.encode(X).shape == (800, 2)
    ae.save(tmp_path / "ae.npz")
    ae2 = Autoencoder.load(tmp_path / "ae.npz")
    assert np.allclose(ae.encode(X), ae2.encode(X))

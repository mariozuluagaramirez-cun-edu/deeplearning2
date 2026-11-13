"""
neural.py
=========
Autoencoder (red neuronal densa) implementado en NumPy para segmentar clientes.

Idea: la red aprende a comprimir cada cliente (5 variables) en un vector latente de 3 números y a
reconstruirlo. La reconstrucción es una versión "depurada" del cliente (filtra el ruido y conserva
la estructura principal); se agrupa con K-Means / Ward / DBSCAN (ver `modeling.py`).

Arquitectura por defecto (5 → 16 → 3 → 16 → 5, 296 parámetros):

    entrada (5) ──Dense 16 + ReLU─▶ capa latente (3, lineal) ─Dense 16 + ReLU─▶ salida (5, lineal)
                  └──────── ENCODER ────────┘                  └────── DECODER ──────┘

¿Por qué NumPy y no PyTorch/Keras? Es una red muy pequeña (296 parámetros): NumPy entrena en segundos,
no añade dependencias pesadas al despliegue en Streamlit Cloud y deja ver explícitamente la
propagación hacia adelante, la retropropagación y el optimizador Adam. El README incluye el código
equivalente en PyTorch.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
AE_PATH = MODELS_DIR / "autoencoder.npz"

NN_FEATURES = ["recency", "frequency", "monetary", "avg_delivery_days", "avg_delay_days"]
LOG_FEATURES = ["recency", "frequency", "monetary", "avg_delivery_days"]


# --------------------------------------------------------------------------- #
# 1. Variables de entrada
# --------------------------------------------------------------------------- #
class FeatureTransformer:
    """Prepara las 5 variables de entrada de la red.

    * `log1p` en Recencia, Frecuencia, Monetario y Días de entrega (asimetría fuerte, todas >= 0).
    * El retraso (`avg_delay_days`) puede ser negativo (llegó antes), así que no admite log: se
      recorta a los percentiles 1 y 99 (winsorización) para que valores extremos (-146 o +189 días)
      no dominen el entrenamiento.
    * `StandardScaler` al final: media 0 y desviación 1, necesario para que la red aprenda bien.
    """

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        self.lo_, self.hi_ = df["avg_delay_days"].quantile([0.01, 0.99]).to_numpy()
        self.scaler_ = StandardScaler().fit(self._pre(df))
        return self

    def _pre(self, df: pd.DataFrame) -> np.ndarray:
        out = df[NN_FEATURES].astype(float).copy()
        for c in LOG_FEATURES:
            out[c] = np.log1p(out[c])
        out["avg_delay_days"] = out["avg_delay_days"].clip(self.lo_, self.hi_)
        return out.to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self._pre(df))


# --------------------------------------------------------------------------- #
# 2. Autoencoder
# --------------------------------------------------------------------------- #
class Autoencoder:
    """Autoencoder denso: n_features → hidden → latent → hidden → n_features (por defecto 5→16→3→16→5)."""

    PARAMS = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]

    def __init__(self, n_features: int = 5, hidden: int = 16, latent: int = 3, seed: int = 42):
        self.n_features, self.hidden, self.latent = n_features, hidden, latent
        rng = np.random.RandomState(seed)
        he = lambda fan_in, fan_out: rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)       # para ReLU
        glorot = lambda fan_in, fan_out: rng.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))  # para lineal
        self.p = {
            "W1": he(n_features, hidden), "b1": np.zeros(hidden),
            "W2": glorot(hidden, latent), "b2": np.zeros(latent),
            "W3": he(latent, hidden), "b3": np.zeros(hidden),
            "W4": glorot(hidden, n_features), "b4": np.zeros(n_features),
        }
        self.history: dict = {}

    # ---- propagación hacia adelante ------------------------------------- #
    def _forward(self, X: np.ndarray):
        p = self.p
        h1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(h1, 0.0)          # capa oculta 1 (ReLU)
        z = a1 @ p["W2"] + p["b2"]                                      # capa latente (lineal)
        h3 = z @ p["W3"] + p["b3"]; a3 = np.maximum(h3, 0.0)          # capa oculta 2 (ReLU)
        xhat = a3 @ p["W4"] + p["b4"]                                   # salida (lineal)
        return h1, a1, z, h3, a3, xhat

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Vector latente (n, latent) de cada cliente."""
        p = self.p
        return np.maximum(X @ p["W1"] + p["b1"], 0.0) @ p["W2"] + p["b2"]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1]

    def loss(self, X: np.ndarray) -> float:
        return float(np.mean((self.reconstruct(X) - X) ** 2))

    # ---- retropropagación ------------------------------------------------ #
    def loss_and_grads(self, X: np.ndarray):
        """Pérdida MSE y gradientes de todos los parámetros (regla de la cadena a mano)."""
        p = self.p
        h1, a1, z, h3, a3, xhat = self._forward(X)
        diff = xhat - X
        loss = float(np.mean(diff ** 2))
        d_xhat = 2.0 * diff / diff.size                    # dL/d(salida)
        g = {"W4": a3.T @ d_xhat, "b4": d_xhat.sum(0)}
        d_h3 = (d_xhat @ p["W4"].T) * (h3 > 0)             # derivada de ReLU
        g["W3"], g["b3"] = z.T @ d_h3, d_h3.sum(0)
        d_z = d_h3 @ p["W3"].T
        g["W2"], g["b2"] = a1.T @ d_z, d_z.sum(0)
        d_h1 = (d_z @ p["W2"].T) * (h1 > 0)
        g["W1"], g["b1"] = X.T @ d_h1, d_h1.sum(0)
        return loss, g

    # ---- entrenamiento --------------------------------------------------- #
    def fit(self, X: np.ndarray, epochs: int = 300, batch_size: int = 256, lr: float = 3e-3,
            val_fraction: float = 0.1, patience: int = 10, min_delta: float = 1e-5,
            seed: int = 42, verbose: int = 0) -> "Autoencoder":
        """Adam + mini-lotes + parada temprana (early stopping) sobre un conjunto de validación.

        Se restauran los pesos de la mejor época de validación, no los de la última.
        """
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        n_val = int(len(X) * val_fraction)
        X_val, X_tr = X[idx[:n_val]], X[idx[n_val:]]

        m = {k: np.zeros_like(v) for k, v in self.p.items()}   # 1er momento de Adam
        v = {k: np.zeros_like(x) for k, x in self.p.items()}   # 2º momento de Adam
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        hist = {"loss": [], "val_loss": []}
        best, best_epoch, best_p, wait = np.inf, 0, None, 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X_tr))
            losses = []
            for s in range(0, len(X_tr), batch_size):
                xb = X_tr[order[s:s + batch_size]]
                loss, g = self.loss_and_grads(xb)
                losses.append(loss)
                t += 1
                for k in self.PARAMS:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
            tr, va = float(np.mean(losses)), self.loss(X_val)
            hist["loss"].append(tr); hist["val_loss"].append(va)
            if verbose and (epoch % verbose == 0 or epoch == 1):
                print(f"época {epoch:3d} | pérdida {tr:.4f} | validación {va:.4f}")
            if va < best - min_delta:
                best, best_epoch, wait = va, epoch, 0
                best_p = {k: x.copy() for k, x in self.p.items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        self.p = best_p                                          # mejores pesos
        hist.update(best_epoch=best_epoch, epochs_run=len(hist["loss"]), best_val_loss=best)
        self.history = hist
        return self

    # ---- utilidades ------------------------------------------------------- #
    @property
    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    def summary(self) -> pd.DataFrame:
        rows = [("Entrada", self.n_features, "—", 0),
                ("Oculta 1 (encoder)", self.hidden, "ReLU", self.p["W1"].size + self.p["b1"].size),
                ("Latente (cuello de botella)", self.latent, "Lineal", self.p["W2"].size + self.p["b2"].size),
                ("Oculta 2 (decoder)", self.hidden, "ReLU", self.p["W3"].size + self.p["b3"].size),
                ("Salida (reconstrucción)", self.n_features, "Lineal", self.p["W4"].size + self.p["b4"].size)]
        return pd.DataFrame(rows, columns=["capa", "neuronas", "activación", "parámetros"]).set_index("capa")

    def save(self, path: Path | str = AE_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        hist = {f"hist_{k}": np.asarray(v) for k, v in self.history.items()}
        np.savez(path, **self.p, **hist)

    @classmethod
    def load(cls, path: Path | str = AE_PATH) -> "Autoencoder":
        d = np.load(path)
        ae = cls(n_features=d["W1"].shape[0], hidden=d["W1"].shape[1], latent=d["W2"].shape[1])
        ae.p = {k: d[k].astype(float) for k in cls.PARAMS}
        ae.history = {k[5:]: (d[k].tolist() if d[k].ndim else d[k].item()) for k in d.files if k.startswith("hist_")}
        return ae


# --------------------------------------------------------------------------- #
# 3. Pipeline neuronal completo
# --------------------------------------------------------------------------- #
class NeuralSpace:
    """Variables → FeatureTransformer → Autoencoder → (reconstrucción | latente estandarizado).

    La reconstrucción (`denoise`) es el espacio donde se aplican K-Means, Ward y DBSCAN; el espacio
    latente estandarizado (`latent`) se usa para visualizar y para explicar la compresión.
    """

    def __init__(self, ft: FeatureTransformer, ae: Autoencoder, zscaler: StandardScaler):
        self.ft, self.ae, self.zscaler = ft, ae, zscaler

    @classmethod
    def train(cls, df: pd.DataFrame, seed: int = 42, verbose: int = 0) -> "NeuralSpace":
        ft = FeatureTransformer().fit(df)
        X = ft.transform(df)
        ae = Autoencoder(n_features=X.shape[1], seed=seed).fit(X, seed=seed, verbose=verbose)
        return cls(ft, ae, StandardScaler().fit(ae.encode(X)))

    def features(self, df: pd.DataFrame) -> np.ndarray:
        """Entrada de la red (n, 5) ya transformada y estandarizada."""
        return self.ft.transform(df)

    def latent(self, X: np.ndarray) -> np.ndarray:
        """Embedding latente estandarizado (n, 3): sirve para VISUALIZAR (no para agrupar, ver nota)."""
        return self.zscaler.transform(self.ae.encode(X))

    def denoise(self, X: np.ndarray) -> np.ndarray:
        """Cliente "depurado": salida de la red (reconstrucción de las 5 variables, n x 5).

        Este es el espacio donde se aplican K-Means / Ward / DBSCAN. Agrupar la reconstrucción
        (que conserva la geometría de las variables originales) resultó MUCHO más estable entre
        entrenamientos que agrupar el espacio latente, cuya geometría cambia con la semilla
        (ver la sección de estabilidad del notebook 02).
        """
        return self.ae.reconstruct(X)

    def save(self, directory: Path | str = MODELS_DIR) -> None:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.ae.save(directory / "autoencoder.npz")
        joblib.dump({"ft": self.ft, "zscaler": self.zscaler}, directory / "neural_preprocessing.joblib")

    @classmethod
    def load(cls, directory: Path | str = MODELS_DIR) -> "NeuralSpace":
        directory = Path(directory)
        pre = joblib.load(directory / "neural_preprocessing.joblib")
        return cls(pre["ft"], Autoencoder.load(directory / "autoencoder.npz"), pre["zscaler"])

    @classmethod
    def load_or_train(cls, df: pd.DataFrame, directory: Path | str = MODELS_DIR) -> "NeuralSpace":
        """Usa los pesos guardados; si no existen o no son compatibles, reentrena (≈10 s)."""
        try:
            return cls.load(directory)
        except Exception:
            return cls.train(df)

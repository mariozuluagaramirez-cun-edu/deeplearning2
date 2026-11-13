"""
modeling.py
===========
Preparación de variables, entrenamiento de 3 algoritmos clásicos de clustering
(K-Means, Jerárquico Ward y DBSCAN), evaluación y perfilamiento de segmentos.

Todas las funciones son puras y reutilizables tanto desde el notebook como
desde la app de Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["recency", "frequency", "monetary"]
MODEL_NAMES = ["K-Means", "Jerárquico (Ward)", "DBSCAN"]

# Hiperparámetros por defecto, justificados con la exploración del notebook
DEFAULTS = {
    "K-Means": {"k": 4},
    "Jerárquico (Ward)": {"k": 4, "sample_size": 8000},
    "DBSCAN": {"eps": 0.15, "min_samples": 200, "sample_size": None},
}


# --------------------------------------------------------------------------- #
# 1. Features
# --------------------------------------------------------------------------- #
def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p (corrige el sesgo fuerte de R, F y M) + estandarización (media 0, desv 1).

    Sin esto, `monetary` (rango de hasta ~13,000) dominaría las distancias euclidianas.
    """
    X_log = np.log1p(df[FEATURES].to_numpy(dtype=float))
    scaler = StandardScaler().fit(X_log)
    return scaler.transform(X_log), scaler


def transform_new(values: pd.DataFrame | np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Aplica al dato de un cliente nuevo la misma transformación del entrenamiento."""
    arr = values[FEATURES].to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, float)
    return scaler.transform(np.log1p(arr))


# --------------------------------------------------------------------------- #
# 2. Modelos
# --------------------------------------------------------------------------- #
@dataclass
class FittedModel:
    name: str
    params: dict
    labels: np.ndarray                       # etiqueta de cada cliente (-1 = ruido en DBSCAN)
    predict: Callable[[np.ndarray], np.ndarray]  # asigna segmento a puntos nuevos (ya escalados)
    artifact: object = None                  # objeto serializable con joblib (para models/)


def fit_kmeans(X: np.ndarray, k: int = 4, seed: int = SEED) -> FittedModel:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return FittedModel("K-Means", {"k": k}, km.labels_, km.predict, artifact=km)


def fit_hierarchical(X: np.ndarray, k: int = 4, sample_size: int = 8000,
                     seed: int = SEED) -> FittedModel:
    """Ward necesita la matriz de distancias (O(n²) en memoria): se ajusta sobre una
    muestra aleatoria y los demás clientes se asignan al centroide más cercano
    (coherente con Ward, que minimiza varianza intra-cluster)."""
    rng = np.random.RandomState(seed)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), n, replace=False)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X[idx])
    nc = NearestCentroid().fit(X[idx], agg.labels_)
    return FittedModel("Jerárquico (Ward)", {"k": k, "sample_size": n},
                       nc.predict(X), nc.predict, artifact=nc)


def fit_dbscan(X: np.ndarray, eps: float = 0.15, min_samples: int = 200,
               sample_size: int | None = None, seed: int = SEED) -> FittedModel:
    """DBSCAN guarda todas las vecindades en memoria (crece ~ n²): con 93k puntos y eps
    grande puede superar varios GB. Si `sample_size` se indica, se ajusta sobre una muestra
    aleatoria con `min_samples` reescalado (misma densidad relativa) y el resto de clientes
    se asigna al core point más cercano a distancia <= eps (definición de punto frontera).
    `min_samples` siempre se interpreta sobre el total de clientes."""
    n = len(X)
    if sample_size is not None and sample_size < n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, sample_size, replace=False)
        ms_eff = max(2, int(round(min_samples * sample_size / n)))
        db = DBSCAN(eps=eps, min_samples=ms_eff).fit(X[idx])
    else:
        idx, ms_eff = None, min_samples
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    cores = db.components_
    core_labels = db.labels_[db.core_sample_indices_]

    def predict(X_new: np.ndarray) -> np.ndarray:
        """Punto nuevo -> cluster del core point más cercano si está a <= eps; si no, ruido (-1)."""
        if len(cores) == 0:
            return np.full(len(X_new), -1)
        dist, ind = NearestNeighbors(n_neighbors=1).fit(cores).kneighbors(X_new)
        out = core_labels[ind[:, 0]].copy()
        out[dist[:, 0] > eps] = -1
        return out

    labels = db.labels_ if idx is None else predict(X)
    return FittedModel("DBSCAN", {"eps": eps, "min_samples": min_samples,
                                  "min_samples_efectivo": ms_eff,
                                  "sample_size": sample_size}, labels, predict,
                       artifact={"core_points": cores, "core_labels": core_labels, "eps": eps})


# Hiperparámetros del enfoque con redes neuronales (clustering sobre la reconstrucción del autoencoder,
# un espacio de 5 dimensiones: por eso `eps` es mayor que en el enfoque clásico de 3 dimensiones)
DEFAULTS_NN = {
    "K-Means": {"k": 5},
    "Jerárquico (Ward)": {"k": 5, "sample_size": 8000},
    "DBSCAN": {"eps": 0.4, "min_samples": 100, "sample_size": 20000},
}


def fit_model(name: str, X: np.ndarray, defaults: dict | None = None, **params) -> FittedModel:
    """Fábrica: entrena el modelo `name` con `params` (usa `defaults` o DEFAULTS para lo no indicado)."""
    p = {**(defaults or DEFAULTS)[name], **params}
    if name == "K-Means":
        return fit_kmeans(X, k=int(p["k"]))
    if name == "Jerárquico (Ward)":
        return fit_hierarchical(X, k=int(p["k"]), sample_size=int(p["sample_size"]))
    if name == "DBSCAN":
        return fit_dbscan(X, eps=float(p["eps"]), min_samples=int(p["min_samples"]),
                          sample_size=p["sample_size"])
    raise ValueError(f"Modelo desconocido: {name}")


# --------------------------------------------------------------------------- #
# 3. Evaluación
# --------------------------------------------------------------------------- #
def evaluate_clustering(X: np.ndarray, labels: np.ndarray, sample_size: int = 10000,
                        seed: int = SEED) -> dict:
    """Silueta (↑), Davies-Bouldin (↓) y Calinski-Harabasz (↑).

    * Los puntos de ruido de DBSCAN (-1) se excluyen del cálculo y se reporta su %.
    * La silueta es O(n²): se calcula sobre una muestra aleatoria fija (reproducible).
    """
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    res = {"n_clusters": n_clusters, "noise_pct": round(float(100 * (~mask).mean()), 2),
           "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if n_clusters < 2:
        return res
    Xm, lm = X[mask], labels[mask]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xm), min(sample_size, len(Xm)), replace=False)
    res["silhouette"] = round(float(silhouette_score(Xm[idx], lm[idx])), 4)
    res["davies_bouldin"] = round(float(davies_bouldin_score(Xm, lm)), 4)
    res["calinski_harabasz"] = round(float(calinski_harabasz_score(Xm, lm)), 1)
    return res


def scan_k(X: np.ndarray, model: str, ks=range(2, 9)) -> pd.DataFrame:
    """Evalúa K-Means o Jerárquico para varios k (insumo para elegir k)."""
    rows = []
    for k in ks:
        m = fit_model(model, X, k=k)
        ev = evaluate_clustering(X, m.labels)
        if model == "K-Means":  # inercia para el método del codo
            ev["inertia"] = round(float(KMeans(k, n_init=10, random_state=SEED).fit(X).inertia_), 1)
        rows.append({"k": k, **ev})
    return pd.DataFrame(rows)


def compare_models(X: np.ndarray, fitted: dict[str, FittedModel]) -> pd.DataFrame:
    rows = [{"modelo": n, **evaluate_clustering(X, m.labels)} for n, m in fitted.items()]
    return pd.DataFrame(rows).set_index("modelo")


# --------------------------------------------------------------------------- #
# 4. Perfilamiento y nombres de segmentos
# --------------------------------------------------------------------------- #
def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Perfil de negocio por segmento (en unidades originales, no escaladas)."""
    d = df.copy()
    d["segment"] = labels
    g = d.groupby("segment")
    prof = pd.DataFrame({
        "clientes": g.size(),
        "recencia_media": g["recency"].mean(),
        "frecuencia_media": g["frequency"].mean(),
        "monetario_medio": g["monetary"].mean(),
        "monetario_mediano": g["monetary"].median(),
        "ingresos_totales": g["monetary"].sum(),
    })
    if "avg_review_score" in d:
        prof["review_promedio"] = g["avg_review_score"].mean()
    if "avg_delivery_days" in d:                       # v2: experiencia de entrega
        prof["entrega_media_dias"] = g["avg_delivery_days"].mean()
        prof["retraso_medio_dias"] = g["avg_delay_days"].mean()
    prof["pct_clientes"] = 100 * prof["clientes"] / prof["clientes"].sum()
    prof["pct_ingresos"] = 100 * prof["ingresos_totales"] / prof["ingresos_totales"].sum()
    return prof.round(2)


def name_segments(prof: pd.DataFrame, df: pd.DataFrame, use_delivery: bool = False) -> dict[int, str]:
    """Nombres de negocio por reglas simples sobre el perfil de cada segmento.

    Referencias globales: mediana de recencia y percentil 75 de monetario (y, si `use_delivery`,
    percentil 75 de los días de entrega). El prefijo con el id garantiza nombres únicos y trazables.
    Con `use_delivery=True` se priorizan los segmentos definidos por la experiencia de entrega.
    """
    rec_med = df["recency"].median()
    mon_p75 = df["monetary"].quantile(0.75)
    ent_p75 = df["avg_delivery_days"].quantile(0.75) if "avg_delivery_days" in df else np.inf
    names = {}
    for seg, r in prof.iterrows():
        if seg == -1:
            names[seg] = "Ruido / atípicos"
            continue
        if use_delivery and r["retraso_medio_dias"] > 0:
            tag = "Entrega tardía"
        elif use_delivery and r["entrega_media_dias"] >= ent_p75:
            tag = "Entrega lenta"
        elif r["pct_clientes"] >= 60:
            tag = "Núcleo masivo (cliente típico)"
        elif r["frecuencia_media"] >= 1.5:
            tag = "Recurrentes"
        else:
            reciente = r["recencia_media"] <= rec_med
            alto = r["monetario_medio"] >= mon_p75
            tag = {(True, True): "Recientes de alto valor",
                   (True, False): "Recientes de ticket bajo",
                   (False, True): "Inactivos de alto valor (en riesgo)",
                   (False, False): "Inactivos de ticket bajo (dormidos)"}[(reciente, alto)]
        names[seg] = f"{int(seg)} · {tag}"
    return names


# --------------------------------------------------------------------------- #
# 5. Acciones de negocio sugeridas por tipo de segmento
# --------------------------------------------------------------------------- #
SEGMENT_ACTIONS = {
    "Recientes de alto valor": "Fidelizar: programa VIP, early access y ventas cruzadas premium.",
    "Recientes de ticket bajo": "Hacer crecer el ticket: bundles, envío gratis sobre un monto y recomendaciones de productos.",
    "Inactivos de alto valor": "Reactivar con prioridad: campañas personalizadas de reconquista y cupones de recompra (concentran gran parte de los ingresos).",
    "Inactivos de ticket bajo": "Reactivación de bajo costo: email automatizado y promociones por categoría favorita; no sobreinvertir.",
    "Entrega tardía": "Prioridad de servicio: sus pedidos llegan después de lo prometido y su reseña es la más baja; compensar (cupón o reembolso de envío) y auditar la logística.",
    "Entrega lenta": "Mejorar la logística: promesas de entrega realistas y transportistas alternativos para evitar que baje su satisfacción.",
    "Recurrentes": "Cuidar y recompensar: son la minoría que ya recompra; programa de puntos y comunicación frecuente.",
    "Núcleo masivo": "Segmento base: acciones masivas y A/B testing; refinar con más variables para subdividirlo.",
    "Ruido": "Revisar caso a caso: clientes atípicos (valores extremos) que pueden ser VIP o anomalías de datos.",
}


def recommend_action(segment_name: str) -> str:
    """Devuelve la acción sugerida según las palabras clave del nombre del segmento."""
    for key, action in SEGMENT_ACTIONS.items():
        if key in segment_name:
            return action
    return "Analizar el perfil del segmento antes de definir una acción."

"""Pruebas ligeras con datos sintéticos (no requieren los CSV de Olist).  Ejecutar: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prep import build_rfm  # noqa: E402
from src.modeling import (MODEL_NAMES, evaluate_clustering, fit_model,  # noqa: E402
                          name_segments, prepare_features, profile_segments, transform_new)


def toy_orders():
    ts = pd.to_datetime(["2018-01-01", "2018-03-01", "2018-02-01"])
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_unique_id": ["A", "A", "B"],
        "customer_state": ["SP", "SP", "RJ"],
        "order_purchase_timestamp": ts,
        "order_value": [100.0, 50.0, 30.0],
        "installments": [1, 2, 3],
        "delivery_days": [10.0, 20.0, 5.0],
        "delay_days": [-5.0, 3.0, -2.0],
    })


def test_rfm_uses_unique_customer_and_aggregates():
    rfm = build_rfm(toy_orders(), reference_date=pd.Timestamp("2018-03-02")).set_index("customer_unique_id")
    assert rfm.loc["A", "frequency"] == 2 and rfm.loc["A", "monetary"] == 150.0
    assert rfm.loc["B", "frequency"] == 1
    assert rfm.loc["A", "avg_delivery_days"] == 15.0 and rfm.loc["A", "avg_delay_days"] == -1.0
    assert rfm.loc["A", "recency"] == 1          # última compra 2018-03-01
    assert rfm.loc["B", "recency"] == 29         # 2018-02-01 -> 2018-03-02


def synthetic_customers(n=1500, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "recency": rng.randint(1, 600, n),
        "frequency": 1 + (rng.rand(n) < 0.05) * rng.randint(1, 4, n),
        "monetary": np.exp(rng.normal(4.7, 0.8, n)),
        "avg_review_score": rng.randint(1, 6, n).astype(float),
    })
    return df


def test_models_return_one_label_per_customer_and_metrics():
    df = synthetic_customers()
    X, scaler = prepare_features(df)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    for name in MODEL_NAMES:
        params = {"sample_size": 800} if name != "K-Means" else {}
        if name == "DBSCAN":
            params.update(eps=0.5, min_samples=10)
        fm = fit_model(name, X, **params)
        assert len(fm.labels) == len(df)
        ev = evaluate_clustering(X, fm.labels)
        assert set(ev) >= {"silhouette", "davies_bouldin", "calinski_harabasz", "noise_pct"}


def test_predict_and_naming():
    df = synthetic_customers()
    X, scaler = prepare_features(df)
    fm = fit_model("K-Means", X, k=3)
    new = pd.DataFrame({"recency": [10], "frequency": [1], "monetary": [100.0]})
    seg = fm.predict(transform_new(new, scaler))
    assert seg.shape == (1,) and 0 <= seg[0] < 3
    prof = profile_segments(df, fm.labels)
    names = name_segments(prof, df)
    assert set(names) == set(prof.index) and len(set(names.values())) == len(names)

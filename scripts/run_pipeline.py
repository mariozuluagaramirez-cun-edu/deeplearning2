"""
run_pipeline.py
===============
Reproduce de punta a punta (sin abrir el notebook):

    CSV crudos de Olist  ->  tabla RFM + tiempos de entrega  ->  3 modelos clásicos
                         ->  autoencoder + 3 modelos sobre su reconstrucción  ->  métricas + artefactos

Uso (desde la raíz del repo):
    python scripts/run_pipeline.py                       # lee data/raw/
    python scripts/run_pipeline.py --raw-dir ruta/a/csv  # otra carpeta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep import PROCESSED_PATH, RAW_DIR, build_customer_dataset  # noqa: E402
from src.modeling import (DEFAULTS_NN, MODEL_NAMES, compare_models, evaluate_clustering,  # noqa: E402
                          fit_model, name_segments, prepare_features, profile_segments)
from src.neural import NeuralSpace  # noqa: E402


def main(raw_dir: Path) -> None:
    # 1) Datos -> RFM
    df = build_customer_dataset(raw_dir)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[ok] {len(df):,} clientes guardados en {PROCESSED_PATH.relative_to(ROOT)}")

    # 2) Features + modelos con hiperparámetros por defecto
    X, scaler = prepare_features(df)
    fitted = {name: fit_model(name, X) for name in MODEL_NAMES}

    # 3) Evaluación
    metrics = compare_models(X, fitted)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    metrics.to_csv(out_dir / "model_comparison.csv")
    print("\n=== Comparación de modelos ===")
    print(metrics.to_string())

    # 4) Perfil de segmentos del mejor modelo (K-Means) y artefactos
    best = fitted["K-Means"]
    prof = profile_segments(df, best.labels)
    prof["nombre"] = prof.index.map(name_segments(prof, df))
    prof.to_csv(out_dir / "kmeans_segment_profile.csv")
    print("\n=== Perfil K-Means ===")
    print(prof.drop(columns=["ingresos_totales"]).to_string())

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    slugs = {"K-Means": "kmeans", "Jerárquico (Ward)": "hierarchical", "DBSCAN": "dbscan"}
    for name, fm in fitted.items():
        joblib.dump(fm.artifact, models_dir / f"{slugs[name]}.joblib")
    print(f"\n[ok] artefactos clásicos guardados en {models_dir.relative_to(ROOT)}/")

    # 5) Enfoque con red neuronal: autoencoder 5->16->3->16->5 + clustering sobre su salida (reconstrucción)
    print("\n=== Red neuronal (autoencoder) ===")
    ns = NeuralSpace.train(df)
    h = ns.ae.history
    print(f"autoencoder: {ns.ae.n_params} parámetros | épocas ejecutadas {h['epochs_run']} | "
          f"mejor época {h['best_epoch']} | MSE validación {h['best_val_loss']:.4f}")
    X5 = ns.features(df)
    Xh = ns.denoise(X5)          # salida de la red: cliente "depurado" (5 variables)
    rows = {}
    nn_models = {name: fit_model(name, Xh, defaults=DEFAULTS_NN) for name in MODEL_NAMES}
    for name, fm in nn_models.items():
        lat, com = evaluate_clustering(Xh, fm.labels), evaluate_clustering(X5, fm.labels)
        rows[f"AE + {name}"] = {"clusters": lat["n_clusters"], "% ruido": lat["noise_pct"],
                                "sil. reconstr.": lat["silhouette"], "DB reconstr.": lat["davies_bouldin"],
                                "CH reconstr.": lat["calinski_harabasz"], "sil. 5 vars": com["silhouette"],
                                "DB 5 vars": com["davies_bouldin"], "CH 5 vars": com["calinski_harabasz"]}
    nn_metrics = pd.DataFrame(rows).T
    nn_metrics.to_csv(out_dir / "model_comparison_nn.csv")
    print(nn_metrics.to_string())
    prof_nn = profile_segments(df, nn_models["K-Means"].labels)
    prof_nn["nombre"] = prof_nn.index.map(name_segments(prof_nn, df, use_delivery=True))
    prof_nn.to_csv(out_dir / "nn_kmeans_segment_profile.csv")
    print(prof_nn[["nombre", "pct_clientes", "entrega_media_dias", "retraso_medio_dias",
                   "review_promedio", "pct_ingresos"]].to_string())
    ns.save(models_dir)
    print(f"\n[ok] autoencoder y preprocesamiento guardados en {models_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Carpeta con los CSV de Olist")
    main(parser.parse_args().raw_dir)

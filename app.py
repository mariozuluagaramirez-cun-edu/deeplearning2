"""
app.py — Segmentación de clientes Olist (RFM + Clustering clásico)
=================================================================
Ejecutar localmente:   streamlit run app.py

La app carga `data/processed/customers_rfm.csv` (generado por el notebook o por
`scripts/run_pipeline.py`), entrena el modelo elegido en segundos y permite explorar
los segmentos, comparar los 3 algoritmos y simular a qué segmento pertenece un cliente.
Dos enfoques: Clásico (RFM) y Red neuronal (autoencoder sobre RFM + tiempos de entrega).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_prep import PROCESSED_PATH, load_processed  # noqa: E402
from src.modeling import (MODEL_NAMES, evaluate_clustering, fit_model,  # noqa: E402
                          name_segments, prepare_features, profile_segments,
                          recommend_action, scan_k, transform_new)
from src.neural import NeuralSpace  # noqa: E402

st.set_page_config(page_title="Segmentación de clientes Olist", page_icon="🛒", layout="wide")

# Parámetros con los que se comparan los modelos en la pestaña "Comparar".
# (Muestras más pequeñas que en el notebook para que la app sea ligera en Streamlit Cloud.)
APP_DEFAULTS = {
    "K-Means": {"k": 4},
    "Jerárquico (Ward)": {"k": 4, "sample_size": 5000},
    "DBSCAN": {"eps": 0.15, "min_samples": 200, "sample_size": 40000},
}
APP_DEFAULTS_NN = {
    "K-Means": {"k": 5},
    "Jerárquico (Ward)": {"k": 5, "sample_size": 5000},
    "DBSCAN": {"eps": 0.4, "min_samples": 100, "sample_size": 15000},
}
VAR_LABELS = {"recency": "Recencia (días)", "frequency": "Frecuencia (nº órdenes)",
              "monetary": "Monetario (R$)", "avg_delivery_days": "Días de entrega",
              "avg_delay_days": "Retraso vs. fecha estimada (días)"}
LOGGABLE = {"recency", "frequency", "monetary", "avg_delivery_days"}   # el retraso puede ser negativo: sin log


# --------------------------------------------------------------------------- #
# Funciones con caché (evitan recalcular en cada interacción)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cargando datos RFM...")
def load_data():
    if not Path(PROCESSED_PATH).exists():
        return None, None, None
    df = load_processed(PROCESSED_PATH)
    X, scaler = prepare_features(df)
    return df, X, scaler


@st.cache_resource(show_spinner="Cargando la red neuronal...")
def load_neural(_df: pd.DataFrame):
    """Autoencoder guardado en models/ (si no existe o no es compatible, se reentrena en ~10 s)."""
    ns = NeuralSpace.load_or_train(_df)
    X5 = ns.features(_df)
    return ns, X5, ns.denoise(X5), ns.latent(X5)


@st.cache_resource(show_spinner="Entrenando modelo...")
def train(name: str, params_key: tuple, space: str, _X: np.ndarray):
    return fit_model(name, _X, **dict(params_key))


@st.cache_data(show_spinner="Calculando métricas...")
def metrics_for(name: str, params_key: tuple, space: str, _X: np.ndarray) -> dict:
    return evaluate_clustering(_X, train(name, params_key, space, _X).labels)


@st.cache_data(show_spinner=False)
def metrics_common(name: str, params_key: tuple, space: str, _X: np.ndarray, _X5: np.ndarray) -> dict:
    """Métricas en el espacio original de 5 variables (común a todos los modelos)."""
    return evaluate_clustering(_X5, train(name, params_key, space, _X).labels)


@st.cache_data(show_spinner="Explorando distintos k (unos segundos)...")
def k_scan(name: str, ks: tuple, space: str, _X: np.ndarray) -> pd.DataFrame:
    return scan_k(_X, name, ks)


@st.cache_data(show_spinner=False)
def pca_coords(space: str, _X: np.ndarray):
    pca = PCA(n_components=2, random_state=42).fit(_X)
    return pca.transform(_X), pca.explained_variance_ratio_


# --------------------------------------------------------------------------- #
# Utilidades de presentación
# --------------------------------------------------------------------------- #
def fmt(x, nd: int = 3) -> str:
    return "N/D" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


def build_color_map(names: dict) -> dict:
    palette = px.colors.qualitative.Safe
    cmap, i = {}, 0
    for seg in sorted(names):
        if seg == -1:
            cmap[names[seg]] = "#9e9e9e"          # ruido en gris
        else:
            cmap[names[seg]] = palette[i % len(palette)]
            i += 1
    return cmap


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
df, X, scaler = load_data()
if df is None:
    st.error("No se encontró `data/processed/customers_rfm.csv`. Ejecuta primero el notebook "
             "o `python scripts/run_pipeline.py --raw-dir <carpeta_con_CSV_de_Olist>`.")
    st.stop()

st.title("🛒 Segmentación de clientes · Olist E-Commerce")
has_delivery = {"avg_delivery_days", "avg_delay_days"} <= set(df.columns)
st.caption("RFM + tiempos de entrega · Machine Learning no supervisado (K-Means · Jerárquico Ward · DBSCAN) · "
           f"{len(df):,} clientes únicos con al menos una compra entregada")

# --------------------------------------------------------------------------- #
# Barra lateral: configuración
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Configuración")
approach = st.sidebar.radio(
    "Enfoque", ["Clásico (RFM)", "Red neuronal (Autoencoder)"] if has_delivery else ["Clásico (RFM)"],
    help="Clásico: agrupa Recencia, Frecuencia y Monetario. Red neuronal: un autoencoder (5→16→3→16→5) "
         "procesa RFM + tiempos de entrega y se agrupa su salida.")
nn = approach.startswith("Red")
model_name = st.sidebar.selectbox(
    "Modelo de clustering", MODEL_NAMES,
    help="K-Means: particiones esféricas. Jerárquico (Ward): agrupa minimizando varianza. "
         "DBSCAN: por densidad, detecta ruido.")

params: dict = {}
if model_name == "K-Means":
    params["k"] = st.sidebar.slider("Número de segmentos (k)", 2, 8, 5 if nn else 4)
elif model_name == "Jerárquico (Ward)":
    params["k"] = st.sidebar.slider("Número de segmentos (k)", 2, 8, 5 if nn else 4)
    params["sample_size"] = st.sidebar.slider(
        "Muestra de ajuste", 2000, 8000, 5000, 1000,
        help="Ward requiere una matriz de distancias O(n²): se ajusta con una muestra y "
             "el resto de clientes se asigna al centroide más cercano.")
else:
    eps_rng = (0.2, 0.8, 0.4, 0.05) if nn else (0.05, 0.20, 0.15, 0.01)
    params["eps"] = round(st.sidebar.slider("eps (radio del vecindario)", *eps_rng,
                                            help="Distancia en el espacio escalado (5 dimensiones con red "
                                                 "neuronal, 3 en el enfoque clásico)."), 2)
    params["min_samples"] = st.sidebar.slider("min_samples", 50, 300 if nn else 500, 100 if nn else 200, 10,
                                              help="Vecinos mínimos (sobre el total de clientes) "
                                                   "para considerar una zona densa.")
    params["sample_size"] = st.sidebar.slider(
        "Muestra de ajuste", 10000, 30000 if nn else 60000, 15000 if nn else 40000, 5000,
        help="DBSCAN consume memoria ~n². Se ajusta con una muestra (min_samples se reescala) "
             "y el resto se asigna al core point más cercano.")

st.sidebar.divider()
all_states = sorted(df["customer_state"].unique())
states = st.sidebar.multiselect("Filtrar por estado (UF)", all_states, default=all_states,
                                help="Filtra lo que se muestra; el modelo se entrena con todos los clientes.")
max_points = st.sidebar.slider("Máx. puntos en gráficos", 2000, 20000, 8000, 1000)

# --------------------------------------------------------------------------- #
# Entrenamiento y segmentación
# --------------------------------------------------------------------------- #
params_key = tuple(sorted(params.items()))
if nn:
    ns, X5, Xh, Z = load_neural(df)
    X_space, space = Xh, "nn"                  # se agrupa la salida del autoencoder (cliente "depurado")
    to_space = lambda new: ns.denoise(ns.features(new))
else:
    X_space, space = X, "rfm"
    to_space = lambda new: transform_new(new, scaler)
fm = train(model_name, params_key, space, X_space)

prof_all = profile_segments(df, fm.labels)
names = name_segments(prof_all, df, use_delivery=nn)
cmap = build_color_map(names)

df_seg = df.copy()
df_seg["cluster"] = fm.labels
df_seg["segmento"] = df_seg["cluster"].map(names)
coords, var_ratio = pca_coords(space, X_space)
df_seg["pc1"], df_seg["pc2"] = coords[:, 0], coords[:, 1]
if nn:
    df_seg["lat1"], df_seg["lat2"], df_seg["lat3"] = Z[:, 0], Z[:, 1], Z[:, 2]

view = df_seg[df_seg["customer_state"].isin(states)]
if view.empty:
    st.warning("Selecciona al menos un estado en la barra lateral.")
    st.stop()
plot_df = view.sample(min(max_points, len(view)), random_state=42)
prof_view = profile_segments(view, view["cluster"].to_numpy())
prof_view["nombre"] = prof_view.index.map(names)

ev = metrics_for(model_name, params_key, space, X_space)
ev5 = metrics_common(model_name, params_key, space, X_space, X5) if nn else None
if ev["n_clusters"] < 2:
    st.warning("Con estos parámetros solo se formó 1 segmento (o ninguno): las métricas no se "
               "pueden calcular. Prueba reduciendo `eps` o `min_samples`.")
var_opts = [v for v in VAR_LABELS if has_delivery or v in ("recency", "frequency", "monetary")]

tab_res, tab_viz, tab_prof, tab_cmp, tab_nn, tab_sim, tab_about = st.tabs(
    ["📊 Resumen", "🧊 Visualización", "🧾 Perfil de segmentos", "⚖️ Comparar modelos",
     "🧠 Red neuronal", "🔮 Simulador", "ℹ️ Acerca de"])

# --------------------------------------------------------------------------- #
# Pestaña 1: Resumen
# --------------------------------------------------------------------------- #
with tab_res:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes (filtro actual)", f"{len(view):,}")
    c2.metric("Ingresos totales", f"R$ {view['monetary'].sum():,.0f}")
    c3.metric("Segmentos encontrados", ev["n_clusters"])
    c4.metric("Clientes como ruido", f"{ev['noise_pct']:.1f}%")

    st.subheader(f"Calidad del agrupamiento · {model_name}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Coeficiente de Silueta ↑", fmt(ev["silhouette"]),
              help="De -1 a 1. Qué tan cerca está cada cliente de su segmento vs. el más cercano. >0.5 buena, 0.25-0.5 aceptable.")
    m2.metric("Davies-Bouldin ↓", fmt(ev["davies_bouldin"]),
              help="Mide solapamiento entre segmentos. Menor es mejor (0 es lo ideal).")
    m3.metric("Calinski-Harabasz ↑", fmt(ev["calinski_harabasz"], 0),
              help="Razón entre dispersión entre-segmentos e intra-segmento. Mayor es mejor.")
    if nn:
        st.caption("Métricas en el espacio que se agrupó (salida de la red, arriba) y en el espacio original de "
                   "5 variables (abajo, comparable con el enfoque clásico).")
        n1, n2, n3 = st.columns(3)
        n1.metric("Silueta · 5 variables ↑", fmt(ev5["silhouette"]))
        n2.metric("Davies-Bouldin · 5 variables ↓", fmt(ev5["davies_bouldin"]))
        n3.metric("Calinski-Harabasz · 5 variables ↑", fmt(ev5["calinski_harabasz"], 0))
    if model_name == "DBSCAN":
        st.info("DBSCAN no fuerza a todos los clientes a un segmento: los puntos en zonas poco densas "
                "se marcan como **ruido** y se excluyen de las métricas. Suele formar un núcleo masivo "
                "más algunos grupos pequeños, porque el RFM de Olist es un continuo sin 'islas' separadas.")

    st.subheader("¿Cuántos clientes y cuánto ingreso concentra cada segmento?")
    share = (view.groupby("segmento")
             .agg(clientes=("monetary", "size"), ingresos=("monetary", "sum")).reset_index())
    share["% de clientes"] = 100 * share["clientes"] / share["clientes"].sum()
    share["% de ingresos"] = 100 * share["ingresos"] / share["ingresos"].sum()
    melt = share.melt(id_vars="segmento", value_vars=["% de clientes", "% de ingresos"],
                      var_name="indicador", value_name="porcentaje")
    fig = px.bar(melt, x="segmento", y="porcentaje", color="indicador", barmode="group",
                 text_auto=".1f", color_discrete_sequence=["#4c78a8", "#f58518"])
    fig.update_layout(xaxis_title="", yaxis_title="%", legend_title_text="", height=420)
    st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Pestaña 2: Visualización
# --------------------------------------------------------------------------- #
with tab_viz:
    modes = ["3D (R-F-M)", "2D (elige ejes)", "PCA 2D"] + (["Espacio latente 3D (red)"] if nn else [])
    mode = st.radio("Tipo de gráfico", modes, horizontal=True)
    log_scale = st.checkbox("Escala logarítmica en los ejes originales", value=True,
                            help="R, F y M son muy sesgadas; el log ayuda a ver la estructura.")
    st.caption(f"Se muestran {len(plot_df):,} clientes (muestra aleatoria) de {len(view):,}. "
               "Haz clic en la leyenda para ocultar/mostrar segmentos.")
    hover = {"customer_state": True, "cluster": False}
    if mode == "3D (R-F-M)":
        fig = px.scatter_3d(plot_df, x="recency", y="frequency", z="monetary", color="segmento",
                            color_discrete_map=cmap, opacity=0.6, labels=VAR_LABELS,
                            log_x=log_scale, log_y=log_scale, log_z=log_scale,
                            hover_data=hover)
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(height=650, legend_title_text="Segmento")
    elif mode == "2D (elige ejes)":
        cx, cy = st.columns(2)
        xvar = cx.selectbox("Eje X", var_opts, index=0, format_func=VAR_LABELS.get)
        yvar = cy.selectbox("Eje Y", var_opts, index=2, format_func=VAR_LABELS.get)
        fig = px.scatter(plot_df, x=xvar, y=yvar, color="segmento", color_discrete_map=cmap,
                         opacity=0.5, labels=VAR_LABELS, log_x=log_scale and xvar in LOGGABLE,
                         log_y=log_scale and yvar in LOGGABLE, hover_data=hover)
        fig.update_layout(height=600, legend_title_text="Segmento")
    elif mode.startswith("Espacio latente"):
        fig = px.scatter_3d(plot_df, x="lat1", y="lat2", z="lat3", color="segmento", color_discrete_map=cmap,
                            opacity=0.6, hover_data=hover,
                            labels={"lat1": "Latente 1", "lat2": "Latente 2", "lat3": "Latente 3"})
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(height=650, legend_title_text="Segmento")
        st.caption("Las 3 neuronas del cuello de botella del autoencoder. Se usa solo para visualizar: "
                   "agrupar directamente este espacio resultó inestable entre entrenamientos (ver pestaña Red neuronal).")
    else:
        fig = px.scatter(plot_df, x="pc1", y="pc2", color="segmento", color_discrete_map=cmap,
                         opacity=0.5, hover_data=hover,
                         labels={"pc1": f"PC1 ({100 * var_ratio[0]:.1f}% var.)",
                                 "pc2": f"PC2 ({100 * var_ratio[1]:.1f}% var.)"})
        fig.update_layout(height=600, legend_title_text="Segmento")
        st.caption("PCA sobre el espacio que agrupa el algoritmo (" + ("salida de la red neuronal, 5 variables"
                   if nn else "RFM transformado: log + estandarizado") + "): proyección 2D de lo que 've'.")
    st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Pestaña 3: Perfil de segmentos
# --------------------------------------------------------------------------- #
with tab_prof:
    st.subheader("Perfil promedio de cada segmento")
    cols = ["nombre", "clientes", "pct_clientes", "recencia_media", "frecuencia_media",
            "monetario_medio"] + (["entrega_media_dias", "retraso_medio_dias"] if has_delivery else []) + \
           ["pct_ingresos", "review_promedio"]
    st.dataframe(prof_view[cols].rename(columns={
        "nombre": "Segmento", "clientes": "Clientes", "pct_clientes": "% clientes",
        "recencia_media": "Recencia media (días)", "frecuencia_media": "Frecuencia media",
        "monetario_medio": "Monetario medio (R$)", "pct_ingresos": "% ingresos",
        "entrega_media_dias": "Días de entrega (media)", "retraso_medio_dias": "Retraso medio (días; <0 = antes)",
        "review_promedio": "Reseña promedio (1-5)"}))

    st.subheader("Distribución de una variable por segmento")
    var = st.selectbox("Variable", var_opts, index=2, format_func=VAR_LABELS.get)
    fig = px.box(plot_df, x="segmento", y=var, color="segmento", color_discrete_map=cmap,
                 labels=VAR_LABELS, log_y=(var in ("recency", "monetary", "avg_delivery_days")))
    fig.update_layout(showlegend=False, xaxis_title="", height=450)
    st.plotly_chart(fig)

    st.subheader("Categorías favoritas del segmento")
    seg_choice = st.selectbox("Segmento", sorted(view["segmento"].unique()))
    top_cat = (view.loc[view["segmento"] == seg_choice, "favorite_category"]
               .value_counts().head(8).rename_axis("categoría").reset_index(name="clientes"))
    fig = px.bar(top_cat.sort_values("clientes"), x="clientes", y="categoría", orientation="h")
    fig.update_layout(height=380, yaxis_title="")
    st.plotly_chart(fig)

    with st.expander("💡 Acciones de marketing sugeridas por segmento", expanded=True):
        for seg in sorted(names):
            st.markdown(f"**{names[seg]}** → {recommend_action(names[seg])}")
        st.caption("Los nombres y acciones se generan con reglas simples sobre el perfil promedio; "
                   "revísalos con criterio de negocio.")

# --------------------------------------------------------------------------- #
# Pestaña 4: Comparar modelos
# --------------------------------------------------------------------------- #
with tab_cmp:
    st.subheader("Los 3 modelos frente a frente")
    st.caption("El modelo seleccionado en la barra lateral usa tus parámetros; los otros usan los valores "
               "por defecto de la app.")
    base_defaults = APP_DEFAULTS_NN if nn else APP_DEFAULTS
    by_model = {n: dict(base_defaults[n]) for n in MODEL_NAMES}
    by_model[model_name] = params
    rows = {n: metrics_for(n, tuple(sorted(p.items())), space, X_space) for n, p in by_model.items()}
    if nn:
        for n, p in by_model.items():
            rows[n]["silueta_5vars"] = metrics_common(n, tuple(sorted(p.items())), space, X_space, X5)["silhouette"]
    cmp_df = pd.DataFrame(rows).T.astype(float).rename(columns={
        "n_clusters": "Nº clusters", "noise_pct": "% ruido", "silhouette": "Silueta ↑",
        "davies_bouldin": "Davies-Bouldin ↓", "calinski_harabasz": "Calinski-Harabasz ↑",
        "silueta_5vars": "Silueta (5 variables) ↑"})
    st.dataframe(cmp_df)

    g1, g2, g3 = st.columns(3)
    for col, metric in zip((g1, g2, g3), ("Silueta ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑")):
        f = px.bar(cmp_df.reset_index(), x="index", y=metric, text_auto=".3s", title=metric,
                   color="index", color_discrete_sequence=px.colors.qualitative.Safe)
        f.update_layout(showlegend=False, xaxis_title="", yaxis_title="", height=320)
        col.plotly_chart(f)
    st.caption("⚠️ Las métricas de DBSCAN se calculan sin los puntos de ruido, así que no son estrictamente "
               "comparables con K-Means/Jerárquico (que clasifican al 100% de los clientes).")

    if model_name in ("K-Means", "Jerárquico (Ward)"):
        if st.checkbox("Mostrar barrido de k (2 a 8) para el modelo seleccionado"):
            scan = k_scan(model_name, tuple(range(2, 9)), space, X_space)
            metrics_to_plot = [("silhouette", "Silueta ↑"), ("davies_bouldin", "Davies-Bouldin ↓"),
                               ("calinski_harabasz", "Calinski-Harabasz ↑")]
            if "inertia" in scan.columns:
                metrics_to_plot.append(("inertia", "Inercia (codo) ↓"))
            for col, (m, title) in zip(st.columns(len(metrics_to_plot)), metrics_to_plot):
                f = px.line(scan, x="k", y=m, markers=True, title=title)
                f.update_layout(height=300, yaxis_title="")
                col.plotly_chart(f)
            st.caption("k=2 suele ganar en silueta porque solo separa a los clientes recurrentes del resto "
                       "(poco accionable). " + ("Con la red neuronal se eligió k=5: maximiza Calinski-Harabasz y "
                       "minimiza Davies-Bouldin." if nn else "Se eligió k=4 por ser el máximo de Calinski-Harabasz y un "
                       "buen equilibrio entre cohesión y utilidad de negocio."))

# --------------------------------------------------------------------------- #
# Pestaña 5: Red neuronal
# --------------------------------------------------------------------------- #
with tab_nn:
    if not nn:
        st.info("Selecciona **Red neuronal (Autoencoder)** en *Enfoque* (barra lateral) para ver la arquitectura, "
                "la curva de aprendizaje y segmentar con la salida de la red.")
    else:
        ae, hist = ns.ae, ns.ae.history
        st.subheader("Autoencoder 5 → 16 → 3 → 16 → 5")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Parámetros entrenables", f"{ae.n_params}")
        a2.metric("Épocas ejecutadas", f"{int(hist.get('epochs_run', 0))}")
        a3.metric("Mejor época", f"{int(hist.get('best_epoch', 0))}")
        a4.metric("MSE de validación", fmt(hist.get("best_val_loss"), 4))
        st.dataframe(ae.summary())
        if hist.get("loss"):
            curve = pd.DataFrame({"época": range(1, len(hist["loss"]) + 1),
                                  "entrenamiento": hist["loss"], "validación": hist["val_loss"]})
            fig = px.line(curve.melt("época", var_name="conjunto", value_name="MSE"), x="época", y="MSE",
                          color="conjunto", title="Curva de aprendizaje (parada temprana en la mejor época)")
            fig.update_layout(height=380, yaxis_range=[0, 0.3])
            st.plotly_chart(fig)
        st.markdown("""
**Cómo funciona.** La red comprime las 5 variables de cada cliente en 3 números (encoder) y las reconstruye
(decoder). Se entrena para que la reconstrucción se parezca a la entrada; así aprende la estructura principal
y descarta ruido. Sobre la **salida reconstruida** se aplican K-Means, Ward y DBSCAN.

| Elemento | Valor | Razón |
|---|---|---|
| Neuronas de entrada | **5** | Recencia, Frecuencia, Monetario, días de entrega y retraso (una neurona por variable) |
| Capas ocultas | **16 + 16**, ReLU | Aportan no linealidad; con 16 el error deja de mejorar de forma apreciable |
| Capa latente | **3**, lineal | Conserva ≈95% de la información (PCA con 3 componentes: ≈74%) y se visualiza en 3D |
| Neuronas de salida | **5**, lineal | Reconstruyen las 5 entradas (variables estandarizadas, sin tope: no sirve sigmoide) |
| Pérdida / optimizador | MSE / Adam (lr 0.003, lote 256) | Regresión de variables continuas; Adam converge de forma robusta |
| Épocas | **máx. 300** con parada temprana (paciencia 10) | El error de validación decide cuándo parar; se restauran los mejores pesos |

⚠️ **Limitación:** agrupar directamente el espacio latente resultó **inestable** entre entrenamientos (ARI medio ≈ 0.18),
mientras que agrupar la salida es estable (ARI ≈ 0.85; K-Means clásico: ≈ 0.97). Por eso la app agrupa la salida y
usa el latente solo para visualizar. Detalles en el notebook 02.
""")

# --------------------------------------------------------------------------- #
# Pestaña 6: Simulador
# --------------------------------------------------------------------------- #
with tab_sim:
    st.subheader("¿A qué segmento pertenece un cliente?")
    st.caption(f"Se usa el modelo seleccionado: **{model_name}** ({approach}). Ingresa el comportamiento del cliente:")
    s1, s2, s3 = st.columns(3)
    r_in = s1.number_input("Recencia (días desde su última compra)", 1, 700, 30)
    f_in = s2.number_input("Frecuencia (nº de órdenes)", 1, 20, 1)
    m_in = s3.number_input("Monetario (R$ gastados en total)", 10.0, 15000.0, 150.0, 10.0)
    d_in, t_in = 10.0, -12.0                                    # valores típicos (solo se piden con la red neuronal)
    if nn:
        s4, s5 = st.columns(2)
        d_in = s4.number_input("Días de entrega (compra → entrega)", 1.0, 120.0, 10.0, 1.0)
        t_in = s5.number_input("Retraso vs. fecha estimada (días; negativo = llegó antes)", -60.0, 60.0, -12.0, 1.0)

    new = pd.DataFrame([{"recency": r_in, "frequency": f_in, "monetary": m_in,
                         "avg_delivery_days": d_in, "avg_delay_days": t_in}])
    seg_id = int(fm.predict(to_space(new))[0])
    seg_name = names.get(seg_id, str(seg_id))
    st.success(f"Segmento asignado: **{seg_name}**")
    st.markdown(f"**Acción sugerida:** {recommend_action(seg_name)}")

    if seg_id in prof_all.index:
        ref = prof_all.loc[seg_id]
        k1, k2, k3 = st.columns(3)
        k1.metric("Recencia: cliente vs. promedio del segmento", f"{r_in} d",
                  f"{r_in - ref['recencia_media']:+.0f} d vs. segmento", delta_color="off")
        k2.metric("Frecuencia", f"{f_in}", f"{f_in - ref['frecuencia_media']:+.2f} vs. segmento",
                  delta_color="off")
        k3.metric("Monetario", f"R$ {m_in:,.0f}", f"R$ {m_in - ref['monetario_medio']:+,.0f} vs. segmento",
                  delta_color="off")
        if nn:
            k4, k5 = st.columns(2)
            k4.metric("Días de entrega", f"{d_in:.0f} d", f"{d_in - ref['entrega_media_dias']:+.1f} d vs. segmento",
                      delta_color="off")
            k5.metric("Retraso", f"{t_in:+.0f} d", f"{t_in - ref['retraso_medio_dias']:+.1f} d vs. segmento",
                      delta_color="off")

# --------------------------------------------------------------------------- #
# Pestaña 7: Acerca de
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("""
### Metodología
1. **Datos:** Brazilian E-Commerce Public Dataset by Olist (Kaggle). Se integran clientes, órdenes, pagos,
   ítems, productos, categorías y reseñas.
2. **Unidad de análisis:** `customer_unique_id` (la persona real; `customer_id` cambia en cada orden).
3. **RFM:** Recencia = días desde la última compra (fecha de referencia = última fecha + 1); Frecuencia = nº de
   órdenes *delivered*; Monetario = suma de `payment_value`.
4. **Tiempos de entrega (v2):** días entre compra y entrega, y retraso frente a la fecha estimada (promedio por cliente).
5. **Enfoque clásico:** `log1p` + `StandardScaler` sobre RFM; K-Means, Ward y DBSCAN.
6. **Enfoque con red neuronal:** un autoencoder 5→16→3→16→5 procesa RFM + entrega; los mismos 3 algoritmos se
   aplican sobre su salida. Métricas: Silueta, Davies-Bouldin y Calinski-Harabasz.

**Hallazgos clave:** ~97% de los clientes compró una sola vez; los inactivos de alto valor concentran la mayor parte
de los ingresos; y los clientes cuyo pedido llegó tarde dan reseñas mucho más bajas (≈2.6 vs. ≈4.3).
""")
    cols_export = ["customer_unique_id", "customer_state", "recency", "frequency", "monetary"] + \
                  (["avg_delivery_days", "avg_delay_days"] if has_delivery else []) + ["segmento"]
    st.download_button("⬇️ Descargar clientes segmentados (CSV)", view[cols_export].to_csv(index=False).encode("utf-8"),
                       file_name="clientes_segmentados.csv", mime="text/csv")

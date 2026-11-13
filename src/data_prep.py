"""
data_prep.py
============
Integración de las tablas de Olist y construcción de la tabla RFM por cliente.

Decisiones metodológicas (documentadas en el README):
  * La unidad de análisis es `customer_unique_id` (la persona real). El campo
    `customer_id` cambia en CADA orden, por lo que usarlo daría Frecuencia = 1 siempre.
  * Solo se consideran órdenes con estado `delivered`: son ventas efectivamente
    concretadas (se excluyen canceladas, no disponibles, en proceso, etc.).
  * Monetario = suma de `payment_value` (lo que el cliente realmente pagó,
    incluye flete). Se agrega primero por orden y luego por cliente.
  * Recencia = días entre la última compra del cliente y una fecha de referencia
    (última fecha del dataset + 1 día).
  * Tiempos de entrega (v2, para el enfoque con redes neuronales):
      - avg_delivery_days = días entre la compra y la entrega al cliente (promedio por cliente).
      - avg_delay_days    = días entre la entrega real y la fecha estimada (negativo = llegó antes
                            de lo prometido; positivo = llegó tarde), promedio por cliente.
    Los 8 pedidos "delivered" sin fecha de entrega quedan como NaN y los clientes sin ningún
    dato se imputan con la mediana global.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "customers_rfm.csv"

# Tablas que SÍ participan en la construcción de RFM y su perfilamiento.
# (sellers y geolocation no aportan al RFM a nivel cliente; ver README.)
REQUIRED_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
}


def load_raw(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Carga los CSV necesarios desde `raw_dir` y devuelve un dict de DataFrames."""
    raw_dir = Path(raw_dir)
    missing = [f for f in REQUIRED_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos en {raw_dir}: {missing}. "
            "Descárgalos de https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
        )
    tables = {name: pd.read_csv(raw_dir / fname) for name, fname in REQUIRED_FILES.items()}
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        tables["orders"][col] = pd.to_datetime(tables["orders"][col])
    return tables


def build_order_table(tables: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """Una fila por orden entregada, con cliente real, fecha y valor pagado."""
    orders, customers, payments = tables["orders"], tables["customers"], tables["payments"]

    # 1) Solo órdenes entregadas
    delivered = orders.loc[orders["order_status"] == "delivered",
                           ["order_id", "customer_id", "order_purchase_timestamp",
                            "order_delivered_customer_date", "order_estimated_delivery_date"]].copy()
    # Tiempos de entrega en días (fracciones de día incluidas)
    one_day = pd.Timedelta(days=1)
    delivered["delivery_days"] = (delivered["order_delivered_customer_date"]
                                  - delivered["order_purchase_timestamp"]) / one_day
    delivered["delay_days"] = (delivered["order_delivered_customer_date"]
                               - delivered["order_estimated_delivery_date"]) / one_day

    # 2) Valor total y cuotas por orden (una orden puede tener varios pagos: cuotas, vouchers...)
    pay_by_order = (
        payments.groupby("order_id")
        .agg(order_value=("payment_value", "sum"),
             installments=("payment_installments", "max"))
        .reset_index()
    )

    # 3) Cliente real (customer_unique_id) + estado
    df = (
        delivered
        .merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
               on="customer_id", how="left", validate="m:1")
        .merge(pay_by_order, on="order_id", how="left", validate="1:1")
    )

    # 4) Limpieza: órdenes sin pago registrado o con valor <= 0 no aportan información monetaria
    n0 = len(df)
    df = df.dropna(subset=["order_value", "customer_unique_id"])
    df = df[df["order_value"] > 0].copy()
    if verbose:
        print(f"[data_prep] órdenes entregadas: {n0:,} -> válidas: {len(df):,}")
    return df


def build_rfm(order_table: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Calcula Recencia, Frecuencia y Monetario por `customer_unique_id`."""
    if reference_date is None:
        reference_date = order_table["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        order_table.sort_values("order_purchase_timestamp")
        .groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_value", "sum"),
            avg_installments=("installments", "mean"),
            customer_state=("customer_state", "last"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_days=("delay_days", "mean"),
        )
        .reset_index()
    )
    for col in ["avg_delivery_days", "avg_delay_days"]:      # clientes sin fecha de entrega
        rfm[col] = rfm[col].fillna(rfm[col].median())
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days.astype(int)
    return rfm.drop(columns="last_purchase")


def add_profile_features(rfm: pd.DataFrame, tables: dict[str, pd.DataFrame],
                         order_table: pd.DataFrame) -> pd.DataFrame:
    """Variables extra SOLO para describir segmentos (no entran al clustering):
    puntaje promedio de reseñas y categoría favorita (en inglés)."""
    # Reseñas: algunas órdenes tienen >1 reseña -> promediamos por orden primero
    rev = (tables["reviews"].groupby("order_id")["review_score"].mean().reset_index()
           .merge(order_table[["order_id", "customer_unique_id"]], on="order_id"))
    avg_review = rev.groupby("customer_unique_id")["review_score"].mean().rename("avg_review_score")

    # Categoría favorita = la más comprada (nº de ítems)
    products = tables["products"][["product_id", "product_category_name"]].merge(
        tables["translation"], on="product_category_name", how="left")
    products["category"] = (products["product_category_name_english"]
                            .fillna(products["product_category_name"]).fillna("unknown"))
    items = (tables["items"][["order_id", "product_id"]]
             .merge(products[["product_id", "category"]], on="product_id", how="left")
             .merge(order_table[["order_id", "customer_unique_id"]], on="order_id"))
    fav = (items.groupby(["customer_unique_id", "category"]).size().reset_index(name="n")
           .sort_values(["customer_unique_id", "n"], ascending=[True, False])
           .drop_duplicates("customer_unique_id")
           .set_index("customer_unique_id")["category"].rename("favorite_category"))

    out = (rfm.merge(avg_review, on="customer_unique_id", how="left")
              .merge(fav, on="customer_unique_id", how="left"))
    out["favorite_category"] = out["favorite_category"].fillna("unknown")
    return out


def build_customer_dataset(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Pipeline completo: CSV crudos -> tabla de clientes con RFM + perfil."""
    tables = load_raw(raw_dir)
    orders = build_order_table(tables)
    rfm = build_rfm(orders)
    cols = ["customer_unique_id", "recency", "frequency", "monetary",
            "avg_installments", "customer_state", "avg_delivery_days", "avg_delay_days"]
    out = add_profile_features(rfm[cols], tables, orders)
    # Redondeo a 6 decimales: así el CSV guardado y los datos recién calculados son idénticos bit a bit.
    # (Sin esto, diferencias de ~1e-13 cambian la trayectoria de entrenamiento de la red neuronal.)
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(6)
    return out


def load_processed(path: Path | str = PROCESSED_PATH) -> pd.DataFrame:
    """Lee la tabla RFM ya procesada (la usa la app de Streamlit)."""
    return pd.read_csv(path)

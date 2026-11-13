# Datos crudos (no incluidos en el repositorio)

Descarga **Brazilian E-Commerce Public Dataset by Olist** desde
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce> y descomprime aquí los 9 CSV:

```
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

Son necesarios solo para **regenerar** `data/processed/customers_rfm.csv` (notebook o
`python scripts/run_pipeline.py`). La app de Streamlit funciona sin ellos porque usa el archivo procesado.

"""
Ingeniería de features por cliente.

Modifica extract_customer_features para agregar las estadísticas que
quieras calcular por cliente. El resultado se persiste en
customer_features.csv y se reutiliza al momento de predecir sobre ítems nuevos
(donde no hay historial de compra).
"""

import os

import numpy as np
import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def extract_customer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features agregadas por cliente a partir del historial de compras.

    Esta función se llama UNA SOLA VEZ sobre los datos de entrenamiento.
    El resultado se guarda en customer_features.csv y se reutiliza en test
    porque el conjunto de test no tiene historial de compra para agregar.

    Parameters
    ----------
    train_df : DataFrame completo de compras de entrenamiento (solo positivos).

    Returns
    -------
    pd.DataFrame con una fila por customer_id.
    """
    df = train_df.copy()

    date_cols = [
        "item_release_date",
        "purchase_timestamp",
        "customer_date_of_birth",
        "customer_signup_date",
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                dayfirst=True,
                errors="coerce",
            )

    #Limpieza
    df["customer_gender"] = (
        df["customer_gender"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["purchase_device"] = (
        df["purchase_device"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["item_category"] = (
        df["item_category"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    group = df.groupby("customer_id")

    today_ts = pd.to_datetime(DATA_COLLECTED_AT)

    # ── Ejemplo: edad del cliente en dias ──────────────────────────────────
    customer_age = (
        today_ts - group["customer_date_of_birth"].first()
    ).dt.days // 365

    # ── Ejemplo: antigüedad en la plataforma en dias ──────────────────────
    customer_tenure_days = (
        today_ts - group["customer_signup_date"].first()
    ).dt.days

    # ── TODO: agrega aquí tus propias features ─────────────────────────────

    # Frecuencia total de compras
    rfm_frequency = group.size()

    # Recencia: días desde última compra
    rfm_recency = (
        today_ts - group["purchase_timestamp"].max()
    ).dt.days

    # Monetary: gasto promedio
    rfm_monetary = group["item_price"].mean()

    # Máximo gasto histórico
    rfm_max_spend = group["item_price"].max()

    # Cantidad de categorías distintas
    rfm_unique_cats = group["item_category"].nunique()

    # Promedio de vistas antes de comprar
    rfm_avg_views = group["customer_item_views"].mean()

    # Ratio de compras calificadas
    rfm_rated_ratio = group["purchase_item_rating"].apply(
        lambda x: x.notna().mean()
    )

    #device mode
    rfm_device_mode = group["purchase_device"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"
    )

    #categorias favoritas top3

    top_categories = (
        df.groupby(["customer_id", "item_category"])
        .size()
        .reset_index(name="count")
    )

    top_categories = top_categories.sort_values(
        ["customer_id", "count"],
        ascending=[True, False]
    )

    top_3 = top_categories.groupby("customer_id").head(3).copy()

    top_3["rank"] = top_3.groupby("customer_id").cumcount() + 1

    top_pivot = top_3.pivot(
        index="customer_id",
        columns="rank",
        values="item_category"
    )

    top_pivot.columns = [
        f"customer_top_{c}_cat"
        for c in top_pivot.columns
    ]

    #frec por categoria

    category_freq = pd.crosstab(
        df["customer_id"],
        df["item_category"]
    )

    category_freq.columns = [
        f"freq_cat_{c}"
        for c in category_freq.columns
    ]

    # ── Construir DataFrame final ───────────────────────────────────────────

    # NOTA: para los features del cliente usa la convencion customer_[FEATURE_NAME] ya que esto facilitará el trabajo del preprocessing
    customer_feat = pd.concat(
        {
            "customer_id": group["customer_id"].first(),
            "customer_gender": group["customer_gender"].first(),

            "customer_age": customer_age,
            "customer_tenure_days": customer_tenure_days,

            "rfm_frequency": rfm_frequency,
            "rfm_recency": rfm_recency,
            "rfm_monetary": rfm_monetary,
            "rfm_max_spend": rfm_max_spend,
            "rfm_unique_cats": rfm_unique_cats,
            "rfm_avg_views": rfm_avg_views,
            "rfm_rated_ratio": rfm_rated_ratio,

            "rfm_device_mode": rfm_device_mode,
        },
        axis=1,
    ).reset_index(drop=True)

    #une top categorias y categorias_freq
    customer_feat = customer_feat.merge(
        top_pivot.reset_index(),
        on="customer_id",
        how="left"
    )

    customer_feat = customer_feat.merge(
        category_freq.reset_index(),
        on="customer_id",
        how="left"
    )

    #rellenar nulos
    customer_feat = customer_feat.fillna({
        "customer_top_1_cat": "unknown",
        "customer_top_2_cat": "unknown",
        "customer_top_3_cat": "unknown",
    })

    freq_cols = [c for c in customer_feat.columns if c.startswith("freq_cat_")]
    customer_feat[freq_cols] = customer_feat[freq_cols].fillna(0)

    # Persistir — read_test_data() carga este archivo en lugar de recomputar
    save_path = os.path.abspath(os.path.join(DATA_DIR, "customer_features.csv"))
    customer_feat.to_csv(save_path, index=False)
    print(f"Customer features saved -> {save_path}")
    return customer_feat

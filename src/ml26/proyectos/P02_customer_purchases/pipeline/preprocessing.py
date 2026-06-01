"""
Capa 2 — Preprocesamiento.

Modifica preprocess() para:
  - Crear features derivadas (e.g. dias desde lanzamiento, match de categoria).
  - Decidir que columnas escalar, codificar o vectorizar.
  - Descartar columnas no disponibles en test (purchase_timestamp, etc.).

No modifiques build_processor — solo define las listas de columnas y las
features derivadas en preprocess().
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def build_processor(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    count_features: list,
    passthrough_features: list = [],
    training: bool = True,
) -> pd.DataFrame:
    """Ajusta o carga un ColumnTransformer y retorna el DataFrame transformado.

    Cuando training=True ajusta el preprocessor y lo guarda en disco.
    Cuando training=False lo carga y aplica — nunca ajustar sobre test.

    Parameters
    ----------
    df                    : DataFrame de entrada (sin label ni IDs).
    numeric_features      : columnas numericas -> StandardScaler.
    categorical_features  : columnas categoricas -> OneHotEncoder.
    count_features         : columnas a aplicar CountVectorizer.
    passthrough_features  : columnas que pasan sin transformar.
    training              : True = fit + save; False = load + transform.
    """
    savepath = Path(os.path.abspath(DATA_DIR)) / "preprocessor.pkl"

    if training:
        text_transformers = [(col, TfidfVectorizer(max_features=100), col) for col in count_features]
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numeric_features),
                ("cat", cat_pipeline, categorical_features),
                *text_transformers,
                ("passthrough", "passthrough", passthrough_features),
            ],
            remainder="drop",
        )
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)

    # CountVectorizer devuelve sparse — convertir a denso para DataFrame uniforme
    if hasattr(processed_array, "toarray"):
        processed_array = processed_array.toarray()

    num_cols = numeric_features
    cat_cols = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            categorical_features
        )
    )
    bow_cols = []
    for col in count_features:
        vocab = preprocessor.named_transformers_[col].get_feature_names_out()
        bow_cols.extend([f"{col}_bow_{t}" for t in vocab])

    return pd.DataFrame(
        processed_array,
        columns=list(num_cols) + cat_cols + bow_cols + passthrough_features,
    )


def preprocess(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Define que features usar y como codificarlas.

    Aqui decides:
      - Que columnas numericas escalar (numeric_features).
      - Que columnas categoricas codificar (categorical_features).
      - Que columnas de texto libre vectorizar (text_features).
      - Que features derivadas crear antes del ColumnTransformer.
      - Que columnas descartar (no disponibles en test o con leakage).

    El flag training se pasa directamente a build_processor — no lo cambies.

    Parameters
    ----------
    df       : DataFrame con todas las features post-merge.
    training : True cuando se llama desde read_train_data (ajusta el
               preprocessor). False cuando se llama desde read_test_data
               (aplica el preprocessor guardado).
    """
    # Columnas a descartar: no disponibles en test o ya integradas en otras features
    drop_raw = [
        "purchase_id",
        "item_release_date",  # conviértela a feature numérica si la necesitas
        "item_img_filename",  # reemplázala por features extraídas de la imagen
    ]

    # ── Features derivadas ─────────────────────────────────────────────────
    df["item_release_date"] = pd.to_datetime(df["item_release_date"], format="mixed")

    # Limpieza de strings
    df["customer_gender"] = (
        df["customer_gender"]
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

    df["item_title"] = (
        df["item_title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # item_days_since_release_cutoff: NO borrar — lo usa split_by_days en training.py
    # para separar train/val sin data leakage. Pasa sin escalar via passthrough.
    df["item_days_since_release_cutoff"] = (
        pd.to_datetime(DATA_COLLECTED_AT) - df["item_release_date"]
    ).dt.days

    # Features del producto
    df["item_days_since_release"] = df["item_days_since_release_cutoff"]
    df["item_release_dayofweek"] = df["item_release_date"].dt.dayofweek
    df["item_release_month"] = df["item_release_date"].dt.month

    # Match con categorías favoritas del cliente
    for i in range(1, 4):
        col = f"customer_top_{i}_cat"
        if col in df.columns:
            df[f"customer_top_{i}_match"] = (
                df[col].astype(str).str.lower().str.strip() == df["item_category"]
            ).astype(int)
        else:
            df[f"customer_top_{i}_match"] = 0

    # Frecuencia del cliente en la categoría del producto
    df["customer_category_frequency"] = df.apply(
        lambda row: row.get(f"freq_cat_{row['item_category']}", 0),
        axis=1,
    )

    # Precio vs historial del cliente
    df["price_ratio_vs_customer_avg"] = df["item_price"] / (df["rfm_monetary"] + 1e-6)
    df["price_diff_vs_customer_avg"] = df["item_price"] - df["rfm_monetary"]

    # Si la categoría ya aparece en el historial del cliente
    df["is_favorite_category"] = (df["customer_category_frequency"] > 0).astype(int)

    # ── TODO: crea aquí tus features derivadas ─────────────────────────────
    # Ejemplos:
    #
    # Días desde lanzamiento (para el modelo, no el split):
    #   df["item_days_since_release"] = df["item_days_since_release_cutoff"]
    #
    # Meses desde lanzamiento:
    #   df["item_months_since_release"] = df["item_days_since_release"] // 30
    #
    # Mes de lanzamiento codificado cíclicamente:
    #   df["item_release_month"] = df["item_release_date"].dt.month
    #   df["item_release_month_sin"] = np.sin(2 * np.pi * df["item_release_month"] / 12)
    #   df["item_release_month_cos"] = np.cos(2 * np.pi * df["item_release_month"] / 12)
    #
    # Match entre categoría del ítem y top categorías del cliente:
    #   for i in range(1, 4):
    #       df[f"customer_top_{i}_match"] = (
    #           df[f"customer_top_{i}_cat"] == df["item_category"]
    #       ).astype(int)

    # ── Definicion de grupos de features ───────────────────────────────────
    # Agrega aquí las columnas que quieras escalar con StandardScaler
    numeric_features = [
        "customer_age",
        "customer_tenure_days",
        "rfm_frequency",
        "rfm_recency",
        "rfm_monetary",
        "rfm_max_spend",
        "rfm_unique_cats",
        "rfm_avg_views",
        "rfm_rated_ratio",
        "item_price",
        "item_days_since_release",
        "item_release_dayofweek",
        "item_release_month",
        "customer_category_frequency",
        "price_ratio_vs_customer_avg",
        "price_diff_vs_customer_avg",
        "is_favorite_category",
        "customer_top_1_match",
        "customer_top_2_match",
        "customer_top_3_match",
        "img_mean_r",
        "img_mean_g",
        "img_mean_b",
    ]

    # Agrega aquí columnas categóricas para OneHotEncoder
    categorical_features = [
        "customer_gender",
        "item_category",
        "rfm_device_mode",
    ]
    # Agrega aquí columnas para CountVectorizer
    count_features = [
        "item_title",
    ]

    # Columnas que pasan sin transformar — item_days_since_release_cutoff es
    # necesario para split_by_days; se dropea antes de model.fit() en training.py.
    passthrough_features = ["item_days_since_release_cutoff"]

    # Tirar columnas de id: NO SIRVEN
    id_cols = ["customer_id", "item_id", "label"]
    cols_to_drop = set(drop_raw) | set(id_cols)
    df_input = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ── Aplicar ColumnTransformer ───────────────────────────────────────────
    return build_processor(
        df_input,
        numeric_features,
        categorical_features,
        count_features,
        passthrough_features,
        training,
    )

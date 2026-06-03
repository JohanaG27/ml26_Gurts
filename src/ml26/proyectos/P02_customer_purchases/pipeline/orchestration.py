"""
Capa 3 — Orquestacion. NO MODIFICAR.

Lee, combina y preprocesa los datos en el orden correcto:
  - read_train_data: calcula features de cliente, genera negativos, ajusta el preprocessor.
  - read_test_data : carga features de cliente pre-calculadas, aplica el preprocessor guardado.

Modificar este archivo puede romper la separacion train/test o causar data leakage.
"""

import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline.features.customer import (
    extract_customer_features,
)
from ml26.proyectos.P02_customer_purchases.pipeline.features.image import (
    extract_image_features,
)
from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    df_to_numeric,
    read_csv,
)
from ml26.proyectos.P02_customer_purchases.pipeline.negatives import (
    gen_final_dataset,
    gen_mixed_negatives,
)
from ml26.proyectos.P02_customer_purchases.pipeline.preprocessing import (
    preprocess,
)


def _add_image_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae features de imagen por ítem y las une al DataFrame principal."""
    img_feat = extract_image_features(df)
    return pd.merge(df, img_feat, on="item_id", how="left")


def _add_customer_features(
    df: pd.DataFrame, customer_feat: pd.DataFrame
) -> pd.DataFrame:
    """Une las features agregadas de cliente al DataFrame principal."""
    raw_cols = {
        "customer_id",
        "customer_gender",
        "customer_date_of_birth",
        "customer_signup_date",
    }
    agg_cols = [c for c in customer_feat.columns if c not in raw_cols]
    return pd.merge(
        df, customer_feat[["customer_id"] + agg_cols], on="customer_id", how="left"
    )

def _split_raw_by_days(df: pd.DataFrame, cutoff_days: int = 60):
    """
    Split temporal antes de calcular customer_features y antes de preprocess.
    Esto evita que validation reciba historial futuro del cliente.
    """
    data = df.copy()

    data["item_release_date"] = pd.to_datetime(
        data["item_release_date"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    cutoff_ts = pd.to_datetime(DATA_COLLECTED_AT)
    days_since_release = (cutoff_ts - data["item_release_date"]).dt.days

    val_mask = days_since_release <= cutoff_days

    train_old = data.loc[~val_mask].copy()
    val_recent = data.loc[val_mask].copy()

    train_old = train_old.sample(frac=1, random_state=42).reset_index(drop=True)
    val_recent = val_recent.reset_index(drop=True)

    return train_old, val_recent


def _preprocess_labeled(df: pd.DataFrame, training: bool):
    """
    Agrega imagen, aplica preprocess y separa X/y.
    """
    data = _add_image_features(df)

    processed = preprocess(data, training=training)
    processed = df_to_numeric(processed)

    processed = pd.concat(
        [processed, data["label"].reset_index(drop=True)],
        axis=1,
    )

    y = processed["label"]

    X = processed.drop(
        columns=[
            "label",
            "customer_id",
            "item_id",
            "item_days_since_release_cutoff",
        ],
        errors="ignore",
    )

    return X, y


def read_train_data(cutoff_days: int = 60):
    """
    Carga y preprocesa los datos de entrenamiento evitando data leakage temporal.

    Flujo leakage-safe:
      1. Cargar train_df con compras positivas.
      2. Separar primero positivos viejos y positivos recientes.
      3. Generar negativos de train usando SOLO positivos viejos.
      4. Generar negativos de validation usando SOLO positivos recientes.
      5. Combinar positivos + negativos por bloque.
      6. Calcular customer_features SOLO con positivos viejos.
      7. Aplicar mismas customer_features a train y validation.
      8. Ajustar preprocessor solo con train y transformar validation.
    """
    # 1. Carga el CSV de compras positivas.
    train_df = read_csv("customer_purchases_train")

    # 2. Split temporal ANTES de generar negativos.
    train_old_pos, val_recent_pos = _split_raw_by_days(
        train_df,
        cutoff_days=cutoff_days,
    )

    # 3. Negativos de entrenamiento: solo usan historial viejo.
    train_negatives = gen_mixed_negatives(train_old_pos, n_per_positive=3)
    train_old = gen_final_dataset(train_old_pos, train_negatives)

    # 4. Negativos de validación: solo usan bloque reciente.
    # Esto mantiene una evaluación artificial, pero sin usar compras futuras
    # para decidir qué pares no deben ser negativos en train.
    val_negatives = gen_mixed_negatives(val_recent_pos, n_per_positive=3)
    val_recent = gen_final_dataset(val_recent_pos, val_negatives)

    # 5. Customer features SOLO con compras reales del train viejo.
    customer_feat = extract_customer_features(train_old_pos)

    # 6. Aplicar mismas customer_features a train y validation.
    train_old = _add_customer_features(train_old, customer_feat)
    val_recent = _add_customer_features(val_recent, customer_feat)

    # 7. Preprocess: fit solo con train, transform con validation.
    X_train, y_train = _preprocess_labeled(train_old, training=True)
    X_val, y_val = _preprocess_labeled(val_recent, training=False)

    return X_train, X_val, y_train, y_val


def read_test_data():
    """Carga y preprocesa el conjunto de test.

    Flujo:
      1. Carga el CSV de test (pares cliente × ítem nuevo, sin historial de compra).
      2. Carga las features de cliente calculadas por read_train_data()
         desde customer_features.csv — no se recalculan porque el historial
         de compras no está disponible en test.
      3. Agrega features de cliente al dataset.
      4. Extrae y agrega features de imagen por ítem.  [opcional]
      5. Aplica preprocess cargando el preprocessor guardado (training=False)
         — nunca se ajusta sobre test para evitar data leakage.

    A diferencia de read_train_data, no hay generación de negativos ni separación de etiquetas: el CSV de test ya contiene los pares a predecir y el label está oculto.

    Returns
    -------
    X : pd.DataFrame -- features del test (sin label).
    """
    # 1. Carga el CSV de test.
    df = read_csv("customer_purchases_test")

    # 2. Carga las features de cliente calculadas por read_train_data()
    customer_feat = read_csv("customer_features")

    # 3. Agrega features de cliente al dataset.
    merged = _add_customer_features(df, customer_feat)

    # 4. Extrae y agrega features de imagen por ítem.
    merged = _add_image_features(merged)

    # 5. Aplica preprocess cargando el preprocessor guardado (training=False)
    processed = preprocess(merged, training=False)
    processed = df_to_numeric(processed)
    processed = processed.drop(
        columns=["customer_id", "item_id", "item_days_since_release_cutoff"],
        errors="ignore",
    )
    return processed

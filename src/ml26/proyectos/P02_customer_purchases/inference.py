import pandas as pd
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from ml26.proyectos.P02_customer_purchases.pipeline import (
    read_test_data,
)
from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    read_csv,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np

from ml26.proyectos.P02_customer_purchases.utils import (
    setup_logger,
)

CURRENT_FILE = Path(__file__).resolve()

RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"


def run_inference(model_path: str, X, ids=None):
    """
    Obtener las predicciones del modelo guardado en model_path para los datos de data_path.
    En su caso, utilicen este archivo para calcular las predicciones de data_test y subir sus resultados a la competencia de kaggle.
    """
    full_path = MODELS_DIR / f"{model_path}/model.pkl"
    #logs
    logger = setup_logger("inference", log_dir=RESULTS_DIR / model_path)
    logger.info(f"Loading model from: {full_path}")
    # Cargar el modelo
    model = joblib.load(full_path)
    #logs2
    logger.info(f"Input shape: {X.shape}")
    logger.info("Generating predictions...")
    # Realizar la inferencia
    #cambios porque segun lo que vi kaggle quiere prob en pred no 0/1

    probs = model.predict_proba(X)[:, 1]

    results = pd.DataFrame(
        {
            "ID": ids if ids is not None else X.index,
            "pred": probs,
        }
    )

    logger.info(f"Generated predictions: {results.shape}")
    return results


def run_inference_random(X, ids=None):
    """
    Clasificador aleatorio como baseline de comparación.

    Genera predicciones al azar con probabilidad uniforme en [0, 1] y umbral
    en 0.5. Un modelo bien entrenado debe superar este baseline en todas las
    métricas; si no lo hace, hay un problema en el pipeline.

    Retorna el mismo esquema que run_inference (ID, pred, proba) para que
    ambos resultados sean comparables directamente.
    """
    probs = np.random.uniform(0, 1, size=len(X))
    preds = (probs >= 0.5).astype(int)
    return pd.DataFrame(
        {"ID": ids if ids is not None else X.index, "pred": preds, "proba": probs}
    )


if __name__ == "__main__":
    test_ids = read_csv("customer_purchases_test")["purchase_id"]
    X = read_test_data()
    #logs3
    main_logger = setup_logger("inference_main", log_dir=RESULTS_DIR)
    main_logger.info("Inference script started")
    main_logger.info(f"Loaded test data: X={X.shape}")
    main_logger.info(f"Loaded test IDs: {len(test_ids)}")

    model_folder = "logistic_lbfgs_1000_20260601_062108"
    model_name = "model.pkl"
    results = run_inference(model_folder, X, ids=test_ids)
    # Guardar predicciones del modelo
    basepath = RESULTS_DIR / model_folder / "predictions.csv"
    basepath.parent.mkdir(exist_ok=True, parents=True)

    results[["ID", "pred"]].to_csv(basepath, index=False)

    main_logger.info(f"Saved Kaggle submission to: {basepath}")
    main_logger.info(f"Submission shape: {results.shape}")
    main_logger.info("Inference script finished")

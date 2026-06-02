import json
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
# Custom
from ml26.proyectos.P02_customer_purchases.model import (
    PurchaseModel,
)
from ml26.proyectos.P02_customer_purchases.utils import (
    setup_logger,
)
from ml26.proyectos.P02_customer_purchases.pipeline import (
    read_train_data,
)


def split_by_days(X, y, cutoff_days=60):
    """
    Separa train y validación usando item_days_since_release_cutoff.

    Las filas con ítems lanzados hace <= cutoff_days van a validación;
    el resto va a entrenamiento. Esto imita la distribución cold-start
    del test (ítems nuevos) mejor que un split aleatorio.

    Parameters
    ----------
    X            : pd.DataFrame con columna item_days_since_release_cutoff.
    y            : pd.Series de etiquetas alineada con X.
    cutoff_days  : ítems lanzados hace <= este número de días van a val.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    if "item_days_since_release_cutoff" not in X.columns:
        raise ValueError("X must contain an 'item_days_since_release_cutoff' column")

    X = X.copy()

    # Split into train/val usando el valor crudo (no escalado)
    val_mask = X["item_days_since_release_cutoff"] <= cutoff_days
    X = X.drop(columns=["item_days_since_release_cutoff"])
    X_val, y_val = X[val_mask], y[val_mask]
    X_train, y_train = X[~val_mask], y[~val_mask]

    # Shuffle training set
    train_idx = X_train.sample(frac=1, random_state=42).index
    X_train, y_train = X_train.loc[train_idx], y_train.loc[train_idx]

    return X_train, X_val, y_train, y_val


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "auc": roc_auc_score(y, y_proba),
        "average_precision": average_precision_score(y, y_proba),
        "f1": f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "classification_report": classification_report(
            y, y_pred, output_dict=True, zero_division=0
        ),
    }


def run_training(X, y, classifier: str):
    # El modelo genera self.name y self.run_dir al inicializarse
    model = PurchaseModel(classifier=classifier)

    # Logger escribe directamente en la carpeta del run
    logger = setup_logger(model.name, log_dir=model.run_dir)
    logger.info(f"Run: {model.name}")
    logger.info(f"Model parameters: {model.get_config()}")

    #Logs nuevos tuneados
    logger.info(f"Full dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Full dataset positives: {y.sum()} | negatives: {(y == 0).sum()}")

    # Separar en entrenamiento y validacion
    X_train, X_val, y_train, y_val = split_by_days(X, y, cutoff_days=60)
    logger.info(f"Split dataset: {len(X_train)} train / {len(X_val)} val")

    #Logs nuevos tuneados 2
    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")

    logger.info(f"Train positives: {y_train.sum()} | negatives: {(y_train == 0).sum()}")
    logger.info(f"Validation positives: {y_val.sum()} | negatives: {(y_val == 0).sum()}")

    # Entrenamiento
    logger.info(f"Starting model training {classifier}...")

    if classifier.lower() in ["xgb", "xgboost"]:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)

    logger.info("Training completed")

    # Evaluacion train vs validation
    logger.info("Running train and validation evaluation...")

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)

    metrics = {
        "train_auc": train_metrics["auc"],
        "val_auc": val_metrics["auc"],

        "train_average_precision": train_metrics["average_precision"],
        "val_average_precision": val_metrics["average_precision"],

        "validation_f1": val_metrics["f1"],
        "validation_precision": val_metrics["precision"],
        "validation_recall": val_metrics["recall"],

        "train_classification_report": train_metrics["classification_report"],
        "validation_classification_report": val_metrics["classification_report"],

        "auc_gap_train_val": train_metrics["auc"] - val_metrics["auc"],
        "average_precision_gap_train_val": (
            train_metrics["average_precision"] - val_metrics["average_precision"]
        ),

        "model_config": model.get_config(),
    }

    logger.info(f"Train AUC-ROC: {train_metrics['auc']:.4f}")
    logger.info(f"Validation AUC-ROC: {val_metrics['auc']:.4f}")
    logger.info(f"AUC gap train-val: {metrics['auc_gap_train_val']:.4f}")

    logger.info(f"Train Average Precision: {train_metrics['average_precision']:.4f}")
    logger.info(f"Validation Average Precision: {val_metrics['average_precision']:.4f}")
    logger.info(
        f"Average Precision gap train-val: "
        f"{metrics['average_precision_gap_train_val']:.4f}"
    )

    logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
    logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
    logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
    logger.info(
        f"Validation classification report: "
        f"{val_metrics['classification_report']}"
    )

    # Guardar modelo y reporte
    logger.info("Saving model...")
    model.save()
    logger.info(f"Model saved to {model.run_dir}")

    report_path = model.run_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {report_path}")

    return model, model.run_dir


if __name__ == "__main__":
    X, y = read_train_data()

    main_logger = setup_logger("training_main")
    main_logger.info("Training script started")
    main_logger.info(f"Loaded training data: X={X.shape}, y={y.shape}")

    models = ["logistic", "rf", "xgb"]
    for model in models:
        run_training(X, y, model)
    
    main_logger.info("Training script finished")

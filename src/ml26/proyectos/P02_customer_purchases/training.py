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


#split by days ya no esta aqui, se hace en orchestration para evitar que validation reciba historial futuro del cliente.


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


def run_training(X_train, X_val, y_train, y_val, classifier: str):
    # El modelo genera self.name y self.run_dir al inicializarse
    model = PurchaseModel(classifier=classifier)

    # Logger escribe directamente en la carpeta del run
    logger = setup_logger(model.name, log_dir=model.run_dir)
    logger.info(f"Run: {model.name}")
    logger.info(f"Model parameters: {model.get_config()}")

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
    main_logger = setup_logger("training_main")
    main_logger.info("Training script started")

    X_train, X_val, y_train, y_val = read_train_data(cutoff_days=60)

    main_logger.info(
        f"Loaded leakage-safe temporal split: "
        f"X_train={X_train.shape}, X_val={X_val.shape}"
    )

    models = ["logistic", "rf", "xgb"]
    for model in models:
        run_training(X_train, X_val, y_train, y_val, model)

    main_logger.info("Training script finished")

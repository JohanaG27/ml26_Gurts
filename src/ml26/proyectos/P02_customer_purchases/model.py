# Data management
from datetime import datetime
from pathlib import Path
import joblib

# ML
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.callback import EarlyStopping
from imblearn.over_sampling import SMOTE

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"

MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, classifier="Logistic", solver="lbfgs", max_iter=1000):
        # Hyperparameters
        self.solver = solver
        self.max_iter = max_iter
        self.model_type = classifier

        # Nombre único del run — usado para la carpeta y el logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{classifier}_{solver}_{max_iter}_{timestamp}"
        self.run_dir = MODELS_DIR / self.name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = self.get_classifier(classifier)
        self.smote = SMOTE(random_state=42)
        # Pipeline
        self.model = Pipeline(
            [
                (
                    "classifier",
                    self.classifier,
                ),
            ]
        )

    def get_classifier(self, name: str, **args):
        name = name.lower()

        if name in ["logistic", "logisticregression", "logistic regression"]:
            return LogisticRegression(
                solver="saga",
                penalty="l1",
                max_iter=3000,
                C=0.1,
                class_weight="balanced",
                random_state=42
            )


        if name in ["rf", "randomforest", "random forest"]:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        if name in ["xgb", "xgboost"]:
            return xgb.XGBClassifier(
                n_estimators=800,
                learning_rate=0.02,
                max_depth=3,
                min_child_weight=10,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=5.0,
                eval_metric="aucpr",
                random_state=42,
                callbacks=[
                    EarlyStopping(
                        rounds=30,
                        save_best=True,
                        maximize=True,
                    )
                ]
            )

        raise ValueError(f"Modelo no soportado: {name}")


    def fit(self, X, y, eval_set=None):
        if self.model_type.lower() in ["xgb", "xgboost"] and eval_set is not None:
            self.model.fit(
                X,
                y,
                classifier__eval_set=eval_set,
                classifier__verbose=False,
            )
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_config(self):
        return {
            "model": self.model_type,
            "solver": self.solver,
            "max_iter": self.max_iter,
        }

    def save(self):
        """
        Guarda el modelo como model.pkl en self.run_dir.
        """
        filepath = self.run_dir / "model.pkl"
        joblib.dump(self, filepath)
        print(f"[MODEL] Saved model to: {filepath}")
        return filepath

    def load(self, filename: str):
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"[MODEL] Loaded model from: {filepath}")
        return model

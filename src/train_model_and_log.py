import logging
import joblib
from datetime import datetime
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Resolved project root (one level above src/)
_PROJECT_ROOT = Path(__file__).parent.parent

TRIAL_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_CV = "heart_disease_cross_validation_runs_"+str(TRIAL_START_TIME)
EXPERIMENT_BEST = "heart_disease_best_model_runs_"+str(TRIAL_START_TIME)
REGISTERED_MODEL_NAME = "heart_disease_pred_model"
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    logger.info("Loading processed dataset from '%s'...", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset loaded. Shape: %s", df.shape)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        "Train/test split: %d train rows, %d test rows.", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [col for col in X.columns if col not in CATEGORICAL_COLS]
    logger.info("Numeric features : %s", numeric_cols)
    logger.info("Categorical features: %s", CATEGORICAL_COLS)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }


def format_run_name(model_name: str, params: dict) -> str:
    param_str = "_".join(
        [f"{k.split('__')[-1]}={v}" for k, v in params.items()]
    )
    return f"{model_name}_{param_str}"


# ---------------------------------------------------------------------------
# Training + MLflow logging
# ---------------------------------------------------------------------------

def train_and_log(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    mlruns_path: Path,
):
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI set to '%s'.", tracking_uri)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
    }

    param_grid = {
        "logistic_regression": {"model__C": [0.1, 1, 10]},
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10],
        },
    }

    for name, model in models.items():
        logger.info("Training model: %s", name)

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        grid = GridSearchCV(
            pipeline,
            param_grid[name],
            cv=3,
            scoring="roc_auc",
            return_train_score=True,
        )
        grid.fit(X_train, y_train)
        results = grid.cv_results_

        # ── Log every hyperparameter combination ──────────────────────────
        mlflow.set_experiment(EXPERIMENT_CV)
        for i in range(len(results["params"])):
            params = results["params"][i]
            run_name = format_run_name(name, params)

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("model_type", name)
                mlflow.log_params(params)
                mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
                mlflow.log_metric("std_test_score", results["std_test_score"][i])
                if "mean_train_score" in results:
                    mlflow.log_metric(
                        "mean_train_score", results["mean_train_score"][i]
                    )

            logger.info(
                "  CV run logged: %s | mean_test_roc_auc=%.4f",
                run_name,
                results["mean_test_score"][i],
            )

        # ── Log the best estimator for this model ─────────────────────────
        mlflow.set_experiment(EXPERIMENT_BEST)
        best_model = grid.best_estimator_

        with mlflow.start_run(run_name=f"{name}_BEST"):
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("stage", "best_model")

            metrics = evaluate(best_model, X_test, y_test)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            mlflow.log_metric("test_precision", metrics["precision"])
            mlflow.log_metric("test_recall", metrics["recall"])
            mlflow.log_metric("test_roc_auc", metrics["roc_auc"])

            mlflow.sklearn.log_model(best_model, "model")

        logger.info(
            "%s BEST → accuracy=%.4f | precision=%.4f | recall=%.4f | roc_auc=%.4f",
            name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["roc_auc"],
        )


# ---------------------------------------------------------------------------
# Model registration + artifact persistence
# ---------------------------------------------------------------------------

def register_best_model(mlruns_path: Path, models_dir: Path):
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_BEST)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_BEST}' not found. "
            "Run train_and_log() before registering."
        )

    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_roc_auc DESC"],
    )

    if not best_runs:
        raise RuntimeError("No runs found in the best-model experiment.")

    winner = best_runs[0]
    model_type = winner.data.tags.get("model_type", "unknown")
    run_id = winner.info.run_id
    roc_auc = winner.data.metrics.get("test_roc_auc")
    params = winner.data.params

    logger.info("Winner model  : %s", model_type)
    logger.info("Run ID        : %s", run_id)
    logger.info("ROC-AUC       : %.4f", roc_auc)
    logger.info("Best params   : %s", params)
    logger.info("Registering as: '%s'", REGISTERED_MODEL_NAME)

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info(
        "MLflow registration successful — name='%s', version=%s",
        mv.name,
        mv.version,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = models_dir / f"{REGISTERED_MODEL_NAME}.pkl"
    winner_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(winner_model, pkl_path)
    logger.info("Model artifact saved to '%s'.", pkl_path.resolve())


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: Path = _PROJECT_ROOT / "data" / "processed" / "heart_clean.csv",
    mlruns_path: Path = _PROJECT_ROOT / "mlruns",
    models_dir: Path = _PROJECT_ROOT / "models",
):
    logger.info("=== Model training pipeline started ===")

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)

    train_and_log(X_train, X_test, y_train, y_test, preprocessor, mlruns_path)
    register_best_model(mlruns_path, models_dir)

    logger.info("=== Model training pipeline finished ===")


if __name__ == "__main__":
    run_pipeline()

"""Entraînement, évaluation et sérialisation du modèle.

Usage :
    uv run python -m sales_prediction.train
    uv run sales-train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from sales_prediction import __version__
from sales_prediction.config import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    PROJECT_ROOT,
    get_model_dir,
    get_reports_dir,
    load_config,
    resolve_path,
)
from sales_prediction.data import file_sha256, load_clean, save_processed, split_xy
from sales_prediction.evaluate import cross_validate_model, regression_metrics
from sales_prediction.features import build_pipeline

logger = logging.getLogger(__name__)


class ModelQualityError(RuntimeError):
    """Le modèle ne passe pas le quality gate : aucun artefact n'est écrit."""


def _git_commit() -> str:
    from_env = os.getenv("GIT_COMMIT")
    if from_env:
        return from_env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def train(
    config: dict[str, Any],
    model_dir: Path | None = None,
    reports_dir: Path | None = None,
    save_processed_data: bool = True,
) -> dict[str, Any]:
    """Pipeline d'entraînement complet. Renvoie les métadonnées du modèle."""
    channels = config["data"]["channels"]
    df = load_clean(config)
    if save_processed_data:
        logger.info("Données nettoyées écrites dans %s", save_processed(df, config))
    X, y = split_xy(df, config)

    # 1) Évaluation honnête : validation croisée + jeu de test mis de côté
    split = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split["test_size"], random_state=split["random_state"]
    )
    cv_metrics = cross_validate_model(build_pipeline(config), X_train, y_train, config)
    holdout_model = build_pipeline(config).fit(X_train, y_train)
    test_metrics = regression_metrics(y_test, holdout_model.predict(X_test))
    logger.info(
        "CV R2=%.4f (+/-%.4f) | test R2=%.4f RMSE=%.3f MAE=%.3f",
        cv_metrics["r2_mean"],
        cv_metrics["r2_std"],
        test_metrics["r2"],
        test_metrics["rmse"],
        test_metrics["mae"],
    )

    min_r2 = config["quality_gate"]["min_r2"]
    if test_metrics["r2"] < min_r2:
        raise ModelQualityError(
            f"R2 test = {test_metrics['r2']:.4f} < seuil {min_r2} : modèle refusé."
        )

    # 2) Modèle de production : réentraîné sur TOUTES les données (jeu de 200 lignes seulement)
    model = build_pipeline(config).fit(X, y)
    regressor = model.named_steps["regressor"]
    raw_path = resolve_path(config["data"]["raw_path"])
    data_hash = file_sha256(raw_path)

    metadata: dict[str, Any] = {
        "model_name": "sales_linear_regression",
        "model_version": f"{__version__}-{data_hash[:8]}",
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "data_sha256": data_hash,
        "n_samples": len(df),
        "target": config["data"]["target"],
        "channels": channels,
        "features": config["features"],
        "engineered_columns": list(regressor.feature_names_in_),
        "coefficients": {
            "intercept": float(regressor.intercept_),
            **{
                n: float(c)
                for n, c in zip(regressor.feature_names_in_, regressor.coef_, strict=True)
            },
        },
        "feature_ranges": {
            ch: {"min": float(X[ch].min()), "max": float(X[ch].max())} for ch in channels
        },
        "metrics": {"cross_validation": cv_metrics, "test": test_metrics},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    model_dir = model_dir or get_model_dir(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILENAME, compress=3)
    (model_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reports_dir = reports_dir or get_reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metadata["metrics"], indent=2), encoding="utf-8"
    )

    logger.info("Artefacts écrits dans %s (version %s)", model_dir, metadata["model_version"])
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de prédiction des ventes.")
    parser.add_argument("--config", type=Path, default=None, help="Chemin du fichier YAML")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

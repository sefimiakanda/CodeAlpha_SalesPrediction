"""Service d'inférence : chargement du modèle sérialisé, prédictions, effets marginaux, optimisation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from sales_prediction import analysis
from sales_prediction.config import METADATA_FILENAME, MODEL_FILENAME

logger = logging.getLogger(__name__)


class ModelService:
    """Enveloppe le pipeline entraîné et ses métadonnées."""

    def __init__(
        self, pipeline: Any, metadata: dict[str, Any], optimizer_step: float = 1.0
    ) -> None:
        self._pipeline = pipeline
        self._metadata = metadata
        self._step = optimizer_step

    @classmethod
    def load(cls, model_dir: Path, optimizer_step: float = 1.0) -> ModelService:
        model_path = model_dir / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}. Lancez `uv run sales-train` (ou `docker build`)."
            )
        meta_path = model_dir / METADATA_FILENAME
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        trained_with = metadata.get("environment", {}).get("scikit_learn")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "scikit-learn %s au chargement vs %s à l'entraînement : réentraînez le modèle.",
                sklearn.__version__,
                trained_with,
            )
        return cls(joblib.load(model_path), metadata, optimizer_step)

    # --- informations ---
    @property
    def version(self) -> str:
        return str(self._metadata.get("model_version", "unknown"))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def channels(self) -> list[str]:
        return list(self._metadata["channels"])

    @property
    def ranges(self) -> dict[str, dict[str, float]]:
        return self._metadata["feature_ranges"]

    # --- API du modèle (compatible avec analysis.py) ---
    def predict(self, budgets: pd.DataFrame) -> Any:
        return self._pipeline.predict(budgets[self.channels])

    def predict_one(self, budget: Mapping[str, float]) -> float:
        return analysis.predict_sales(self, budget)

    def effects(self, budget: Mapping[str, float]) -> dict[str, float]:
        return analysis.marginal_effects(self, budget)

    def warnings(self, budget: Mapping[str, float]) -> list[str]:
        return analysis.out_of_range(budget, self.ranges)

    def optimize(self, total: float) -> dict[str, Any]:
        upper = {ch: self.ranges[ch]["max"] for ch in self.channels}
        return analysis.optimize_budget(self, total, self.channels, upper, self._step)

    def predict_many(self, budgets: Sequence[Mapping[str, float]]) -> list[float]:
        return [float(v) for v in self.predict(pd.DataFrame([dict(b) for b in budgets]))]

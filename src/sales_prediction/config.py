"""Chargement de la configuration YAML et résolution des chemins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lit configs/config.yaml (ou le fichier indiqué par `path` / la variable CONFIG_PATH)."""
    config_path = Path(path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path) -> Path:
    """Chemin relatif -> chemin absolu depuis la racine du projet."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_model_dir(config: dict[str, Any]) -> Path:
    """Dossier des artefacts ; la variable d'environnement MODEL_DIR prime sur le YAML."""
    return resolve_path(os.getenv("MODEL_DIR") or config["paths"]["model_dir"])


def get_reports_dir(config: dict[str, Any]) -> Path:
    return resolve_path(os.getenv("REPORTS_DIR") or config["paths"]["reports_dir"])

"""Chargement, validation et nettoyage des données brutes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from sales_prediction.config import resolve_path


def load_raw(config: dict[str, Any]) -> pd.DataFrame:
    """Lit le CSV brut tel quel (aucune transformation)."""
    return pd.read_csv(resolve_path(config["data"]["raw_path"]))


def validate(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Vérifie le contrat des données. Lève ValueError au premier problème rencontré."""
    required = [*config["data"]["channels"], config["data"]["target"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La colonne '{col}' doit être numérique")
    if df[required].isna().any().any():
        raise ValueError("Valeurs manquantes détectées")
    if (df[required] < 0).any().any():
        raise ValueError("Valeurs négatives détectées (budgets et ventes doivent être >= 0)")


def clean(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Supprime la colonne d'index, les doublons, puis valide. Renvoie une copie."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    out = out.drop(columns=config["data"].get("drop_columns", []), errors="ignore")
    out = out.drop_duplicates().reset_index(drop=True)
    validate(out, config)
    return out


def load_clean(config: dict[str, Any]) -> pd.DataFrame:
    """Raccourci : lecture + nettoyage."""
    return clean(load_raw(config), config)


def save_processed(df: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Écrit les données nettoyées dans data/processed/."""
    path = resolve_path(config["data"]["processed_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def split_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables explicatives (budgets) et la cible (ventes)."""
    return df[config["data"]["channels"]].copy(), df[config["data"]["target"]].copy()


def file_sha256(path: Path) -> str:
    """Empreinte du fichier : lie chaque modèle à la version exacte des données."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

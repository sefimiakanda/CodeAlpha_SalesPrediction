"""Contrats d'entrée/sortie de l'API (validation Pydantic)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MAX_CHANNEL = 1000.0
MAX_TOTAL = 3000.0


class Budget(BaseModel):
    """Budgets publicitaires par canal, en milliers de dollars."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"tv": 150, "radio": 25, "newspaper": 30}]},
    )

    tv: float = Field(ge=0, le=MAX_CHANNEL, description="Budget TV (k$)")
    radio: float = Field(ge=0, le=MAX_CHANNEL, description="Budget radio (k$)")
    newspaper: float = Field(
        default=0.0, ge=0, le=MAX_CHANNEL, description="Budget presse (k$) : sans effet mesurable"
    )

    def to_record(self) -> dict[str, float]:
        """Noms de colonnes du jeu de données d'entraînement."""
        return {"TV": self.tv, "Radio": self.radio, "Newspaper": self.newspaper}


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_sales: float = Field(description="Ventes prédites (milliers d'unités)")
    marginal_effects: dict[str, float] = Field(
        description="Ventes supplémentaires pour +1 k$ sur chaque canal (les autres restent fixes)"
    )
    warnings: list[str]
    model_version: str


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_budget: float = Field(gt=0, le=MAX_TOTAL, description="Budget total à répartir (k$)")
    current: Budget | None = Field(
        default=None, description="Répartition actuelle, pour mesurer le gain de l'optimisation"
    )


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    allocation: dict[str, float]
    predicted_sales: float
    current_predicted_sales: float | None = None
    uplift: float | None = None
    uplift_pct: float | None = None
    warnings: list[str]
    model_version: str


class HealthResponse(BaseModel):
    status: str


ModelInfo = dict[str, Any]

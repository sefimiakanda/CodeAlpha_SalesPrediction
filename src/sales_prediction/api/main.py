"""Application FastAPI (factory).

Lancement local :
    uv run uvicorn sales_prediction.api.main:create_app --factory --reload
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse

from sales_prediction import __version__
from sales_prediction.api.schemas import (
    Budget,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    PredictionResponse,
)
from sales_prediction.config import get_model_dir, load_config
from sales_prediction.predict import ModelService

logger = logging.getLogger("sales_prediction.api")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class _JsonFormatter(logging.Formatter):
    """Une ligne JSON par log : facile à filtrer dans le dashboard de Render."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def create_app(model_service: ModelService | None = None) -> FastAPI:
    """Construit l'application. `model_service` permet d'injecter un modèle (tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _setup_logging()
        if model_service is not None:
            app.state.model = model_service
        else:
            config = load_config()
            # Fail fast : sans modèle, le conteneur s'arrête au démarrage au lieu de rester « sain » à tort
            app.state.model = ModelService.load(get_model_dir(config), config["optimizer"]["step"])
        logger.info("Modèle chargé", extra={"fields": {"model_version": app.state.model.version}})
        yield

    app = FastAPI(
        title="Sales Prediction API",
        description="Prévoit les ventes à partir des budgets publicitaires et recommande une répartition.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_logging(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logger.info(
                "request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                },
            )

    def model_of(request: Request) -> ModelService:
        return request.app.state.model

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness : le process répond (utilisé par Render et Docker)."""
        return HealthResponse(status="ok")

    @app.get("/model/info", tags=["model"])
    def model_info(request: Request) -> dict[str, Any]:
        """Version, métriques, coefficients et plages de données du modèle en production."""
        return model_of(request).metadata

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(budget: Budget, request: Request) -> PredictionResponse:
        """Prévoit les ventes et l'effet marginal de chaque canal pour un budget donné."""
        model = model_of(request)
        record = budget.to_record()
        return PredictionResponse(
            predicted_sales=round(model.predict_one(record), 3),
            marginal_effects={k: round(v, 4) for k, v in model.effects(record).items()},
            warnings=model.warnings(record),
            model_version=model.version,
        )

    @app.post("/optimize", response_model=OptimizeResponse, tags=["prediction"])
    def optimize(payload: OptimizeRequest, request: Request) -> OptimizeResponse:
        """Répartition optimale d'un budget total, bornée aux plages observées à l'entraînement."""
        model = model_of(request)
        try:
            best = model.optimize(payload.total_budget)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

        current_sales = uplift = uplift_pct = None
        if payload.current is not None:
            current_sales = model.predict_one(payload.current.to_record())
            uplift = best["predicted_sales"] - current_sales
            uplift_pct = uplift / current_sales * 100 if current_sales else None
        return OptimizeResponse(
            allocation={k: round(v, 2) for k, v in best["allocation"].items()},
            predicted_sales=round(best["predicted_sales"], 3),
            current_predicted_sales=None if current_sales is None else round(current_sales, 3),
            uplift=None if uplift is None else round(uplift, 3),
            uplift_pct=None if uplift_pct is None else round(uplift_pct, 1),
            warnings=[
                "Les budgets suggérés sont plafonnés aux maxima observés (TV, Radio, Presse) : "
                "au-delà, le modèle n'a aucune donnée pour juger."
            ],
            model_version=model.version,
        )

    return app

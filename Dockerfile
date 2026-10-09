# Image de production. Corrections par rapport au projet 1 :
#  - Python 3.12 fixé (identique à .python-version et pyproject.toml)
#  - pas de compilation de bytecode (rallongeait inutilement le build)
#  - timeout réseau uv augmenté (téléchargements lents/instables)
#  - .dockerignore présent (le contexte n'embarque plus .venv)
#  - `uv sync --locked` : le build échoue immédiatement si uv.lock n'est pas à jour

# ---------- Étape 1 : build (dépendances + entraînement du modèle) ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=120 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Couche mise en cache : ne se refait que si pyproject.toml / uv.lock changent
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY configs ./configs
COPY data/raw ./data/raw
COPY src ./src
RUN uv sync --locked --no-dev

# Entraînement PENDANT le build : modèle et version de scikit-learn sont garantis identiques.
# Si le R² est sous le seuil de configs/config.yaml, le build échoue (aucun mauvais modèle livré).
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
RUN /app/.venv/bin/python -m sales_prediction.train

# ---------- Étape 2 : runtime (image minimale, sans droits root) ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

RUN useradd --system --create-home app
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/models ./models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=3)"

# $PORT est injecté par Render ; 8000 par défaut en local
CMD ["sh", "-c", "exec uvicorn sales_prediction.api.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]

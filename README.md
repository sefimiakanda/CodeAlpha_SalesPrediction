# 📈 Sales Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-purple)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)
![CI](https://github.com/sefimiakanda/CodeAlpha_SalesPrediction/actions/workflows/ci.yml/badge.svg)

Forecast sales from advertising budgets, measure the impact of each channel, and recommend a better budget split — from exploratory notebook to a Dockerized API deployed on Render.

🔗 **Live demo:** `https://<your-service>.onrender.com` *(replace with your Render URL; the free tier may take about a minute to wake up)*

## 🎯 Project Overview

Developed as part of an internship project at CodeAlpha. The brief:

| Objective | How it is addressed |
|---|---|
| Predict future sales from advertising spend, segment and platform | Regression on the three platforms (TV, Radio, Newspaper); `POST /predict` forecasts sales for any planned budget |
| Clean, transform, select features | Notebook sections 3, 8 and 9; production code in `data.py` and `features.py` |
| Regression or time series | **Regression.** The dataset has no date column, so a time-series model is not applicable; it also has no customer-segment column |
| Analyze how advertising changes sales | Marginal effects, response curves, what-if scenarios, budget optimizer (`analysis.py`, notebook section 12) |
| Actionable marketing insights | Recommendations at the end of the notebook; `POST /optimize` and the web interface |

## 🔎 Key Findings

*(Dataset: 200 campaigns, budgets in thousands of dollars, sales in thousands of units.)*

- **Newspaper has no measurable effect.** Its p-value is ≈ 0.70, it adds nothing in cross-validation, and its raw correlation with sales comes from its link with Radio spend (a confounder). It is excluded from the model.
- **TV and Radio reinforce each other** (positive TV × Radio interaction), and **TV shows diminishing returns** (negative TV² term). A model with these two terms reaches a cross-validated **R² ≈ 0.98**, against ≈ 0.88 for a plain linear regression, and matches or beats Random Forest and Gradient Boosting while staying interpretable.
- **At the average budget, an extra $1,000 on Radio sells about 4× more units than on TV** (≈ 193 vs ≈ 44 units); on Newspaper it sells none.
- **Reallocating the same total budget** (≈ $201k) from the current average mix to TV ≈ $151k + Radio ≈ $49k + no Newspaper raises forecast sales by about **+36 %**. Suggested budgets are capped at the maximum values seen in the data; beyond them the model cannot judge.

Final model on the held-out test set: **R² ≈ 0.989, RMSE ≈ 0.60, MAE ≈ 0.50** (thousand units).

## 📂 Structure

```
CodeAlpha_SalesPrediction/
├── .github/workflows/        # CI (lint, tests, Docker smoke test) and image release
├── configs/config.yaml       # Features, split, CV, quality gate, optimizer: change behaviour without code
├── data/
│   ├── raw/Advertising.csv   # Original data (never edited)
│   └── processed/            # Cleaned data (generated)
├── models/                   # Model + metadata (generated, not versioned)
├── notebooks/
│   └── 01_sales_prediction_eda_modeling.ipynb   # Step-by-step EDA, statistics, modeling, impact analysis
├── src/sales_prediction/
│   ├── config.py             # YAML loading, paths
│   ├── data.py               # Load, validate, clean
│   ├── features.py           # Feature engineering + scikit-learn pipeline
│   ├── evaluate.py           # Metrics, repeated cross-validation
│   ├── analysis.py           # OLS statistics, marginal effects, what-if, budget optimizer
│   ├── train.py              # Training CLI with quality gate
│   ├── predict.py            # Model loading and inference service
│   └── api/                  # FastAPI app, schemas, web interface (static/index.html)
├── tests/                    # Unit and integration tests
├── Dockerfile · .dockerignore
├── render.yaml               # Render Blueprint (Docker deployment)
├── pyproject.toml · uv.lock · .python-version
└── README.md
```

## 🚀 Local Development

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. The project pins **Python 3.12** (`.python-version` and `pyproject.toml`); uv downloads it automatically if needed.

```bash
uv python install 3.12                 # make sure Python 3.12 is available (one-time)
uv sync                                # create .venv, install dependencies, generate uv.lock (commit it)
uv run sales-train                     # train the model -> models/ (and clean data -> data/processed/)
uv run pytest                          # run the test suite
uv run uvicorn sales_prediction.api.main:create_app --factory --reload   # API on http://localhost:8000
```

Open **http://localhost:8000** for the web interface or **http://localhost:8000/docs** for the Swagger UI.

### Notebook

```bash
uv sync --group notebook               # adds JupyterLab, matplotlib, seaborn
uv run jupyter lab                     # then open notebooks/01_sales_prediction_eda_modeling.ipynb
```

Run the cells one at a time and read the comment and result of each step. The notebook reuses the production functions from `src/`, so what is validated there is what is deployed.

### Code quality

```bash
uv run ruff check --fix .              # lint (and sort imports)
uv run ruff format .                   # format
```

## 🔌 API

| Endpoint | Description |
|---|---|
| `GET /` | Web interface (sliders, live forecast, budget optimizer) |
| `POST /predict` | Forecast sales for `{tv, radio, newspaper}` budgets, with the marginal effect of each channel |
| `POST /optimize` | Best split of a total budget, optionally compared with the current split |
| `GET /model/info` | Model version, metrics, coefficients and the data ranges seen in training |
| `GET /health` | Liveness probe (used by Docker and Render) |
| `GET /docs` | Interactive documentation |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tv": 150, "radio": 25, "newspaper": 30}'

curl -X POST http://localhost:8000/optimize -H "Content-Type: application/json" \
  -d '{"total_budget": 205, "current": {"tv": 150, "radio": 25, "newspaper": 30}}'
```

On Windows PowerShell, use `curl.exe` instead of `curl`, or use the web interface.

Inputs are validated (negative or absurd budgets return HTTP 422). Budgets outside the range seen in training still get a forecast, but with an explicit extrapolation warning.

## 🐳 Docker

```bash
uv lock                                          # once: the build needs uv.lock
docker build --progress=plain -t sales-prediction .
docker run --rm -p 8000:8000 sales-prediction    # http://localhost:8000
```

The image is built in two stages: dependencies are installed with `uv sync --locked`, then **the model is trained during the build** (so the model and the scikit-learn version that loads it are always identical). If the test R² falls below the quality gate in `configs/config.yaml`, the build fails. The final image runs as a non-root user.

Tips if the build is slow or stuck:

- Keep the project **outside OneDrive** (for example `C:\dev\...`): synchronization makes Docker very slow.
- Make sure `.dockerignore` is present (otherwise `.venv` is sent to Docker on every build).
- Check that Docker Desktop is running (`docker info`), restart it with `wsl --shutdown` if it freezes, and test connectivity with  
  `docker run --rm python:3.12-slim python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org/simple/', timeout=15).status)"`
- Slow download? `UV_HTTP_TIMEOUT` is already raised in the `Dockerfile`; a VPN or antivirus can also block it.

## ⚙️ CI/CD

- **`ci.yml`** — on every push and pull request: install from the lockfile, `ruff` (lint and format), tests with coverage, then build the Docker image (which trains the model) and smoke-test the running container.
- **`release.yml`** — on a version tag (`git tag v0.1.0 && git push --tags`): builds and publishes the image to GitHub Container Registry (`ghcr.io`).

## ☁️ Deployment on Render (Docker)

Deployment is described by `render.yaml`; Render builds the `Dockerfile` on its own servers.

```yaml
services:
  - type: web
    name: sales-prediction-api
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    autoDeploy: true
```

**Steps**

1. Make sure `uv.lock`, `.python-version`, `configs/`, `data/raw/Advertising.csv` and `render.yaml` are committed, and push to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com): **New + → Blueprint**, connect GitHub, select the repository, then **Apply**.
3. Follow the **Logs** tab: you should see the dependency installation, the training metrics (`CV R2=… | test R2=…`), then `Uvicorn running`.
4. When the status is **Live**, open `https://sales-prediction-api-xxxx.onrender.com`.

Every `git push` to the linked branch redeploys automatically. To deploy only after CI passes, set *Auto-Deploy → After CI Checks Pass* in the service settings.

**Without Docker:** `render.yaml` contains a commented native-Python variant (`runtime: python`, build and start commands with uv). Swap the blocks to use it.

**Free tier limitations** (they change regularly; check Render's pricing page): the service spins down after about 15 minutes of inactivity and takes around a minute to wake up; memory is limited to about 512 MB; the filesystem is ephemeral (the model is rebuilt on each deployment, so this is not an issue); the URL is public and unauthenticated.

**Troubleshooting**

| Symptom in the logs | Likely cause | Fix |
|---|---|---|
| `uv.lock not found` | The lockfile is not on GitHub | `git add uv.lock`, commit, push |
| `The lockfile needs to be updated` | `pyproject.toml` changed without `uv lock` | Run `uv lock`, commit both files |
| `COPY data/raw ... not found` | The dataset is not committed (or git-ignored) | `git add data/raw/Advertising.csv` |
| `ModelQualityError` | Test R² below `quality_gate.min_r2` | Check the data and `configs/config.yaml` |
| `No open ports detected` | The app does not listen on `$PORT` | Keep the `CMD` of the `Dockerfile` unchanged |

## ⚠️ Limitations

- **Small dataset** (200 campaigns, no date): no seasonality, trend or delayed advertising effect can be modeled.
- **Correlation, not causation:** results describe observed associations; a controlled experiment would confirm them.
- **Extrapolation:** forecasts outside the observed budgets (TV > 296, Radio > 50, Newspaper > 114) are unreliable, and the optimizer is capped at those maxima.
- **Very low TV budgets:** the model overestimates sales when TV spend is close to zero (two such campaigns in the data); do not use it to simulate dropping TV entirely.
- **Statistical caveat:** residuals are not perfectly normal because of those two campaigns, so p-values and confidence intervals are indicative; the API therefore returns no prediction intervals.
- The model maximizes **sales**, not profit: costs and margins are not included.
- Units are those of the original dataset (budgets in thousands of dollars, sales in thousands of units).

## 💎 Production Practices

1. **Config-driven**: features, split, CV and quality gate live in `configs/config.yaml`.
2. **Self-contained pipeline**: feature selection and engineering are inside the saved model, so the API cannot skip them.
3. **Locked dependencies and a pinned Python version** (`uv.lock`, `.python-version`) for identical environments everywhere.
4. **Quality gate**: training (and the Docker build) fails if the model is not good enough.
5. **Strict input validation** with Pydantic, explicit extrapolation warnings, structured JSON logs with request IDs.
6. **Automated tests and CI**, reproducible Docker image, deployment as code with `render.yaml`.

## 👤 Author

**Fidèle Miakanda** — *Data Science Intern at CodeAlpha*
  * [GitHub](https://github.com/sefimiakanda)
  * [LinkedIn](https://www.linkedin.com/in/fidèle-miakanda)

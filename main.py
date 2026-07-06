from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
from pathlib import Path

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


# =========================
# IMPORT ROUTERS
# =========================
from routes import login, patients, reports, diagnostics
from dashboard import router as dashboard_router


# =========================
# APPLICATION
# =========================
app = FastAPI(
    title="ONCO AI API",
    description="API intelligente d'aide au diagnostic médical",
    version="1.0.0"
)


# =========================
# STATIC FILES
# =========================

# Images patients / résultats
app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)


# Fichiers CSS, JS, images du dashboard
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# API ROUTES
# =========================

app.include_router(login.router)
app.include_router(patients.router)
app.include_router(diagnostics.router)
app.include_router(reports.router)


# =========================
# DASHBOARD WEB
# =========================
app.include_router(dashboard_router)


# =========================
# TEST API
# =========================
@app.get("/health")
def health():
    return {
        "status": "ONCO AI API opérationnelle"
    }
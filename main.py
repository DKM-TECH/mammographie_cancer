from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
from pathlib import Path

import os

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


# =========================
# IMPORT ROUTERS
# =========================
from routes import login, patients,  diagnostics
#from dashboard import router as dashboard_router


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

UPLOADS_DIR = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOADS_DIR)),
    name="uploads"
)

# =========================
# CORS
# =========================
#from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://polite-medovik-ed6a00.netlify.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null"
],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# API ROUTES
# =========================

app.include_router(login.router)
app.include_router(patients.router)
app.include_router(diagnostics.router)
#app.include_router(reports.router)


# =========================
# DASHBOARD WEB
# =========================
#app.include_router(dashboard_router)


# =========================
# TEST API
# =========================
@app.get("/health")
def health():
    return {
        "status": "ONCO AI API opérationnelle"
    }

@app.get("/")
def root():
    return {
        "application": "ONCO AI",
        "message": "API d'aide à l'interprétation des mammographies",
        "status": "running"
    }
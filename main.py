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
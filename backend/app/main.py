"""
FastAPI Backend Application Entry Point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import api_router
from backend.app.database.db import init_db
from backend.app.services.ml_service import ml_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database and Warmup Models
    print("[CyberAttack-IDS] Starting backend services & database...")
    init_db()
    ml_service._initialize()
    print(f"[CyberAttack-IDS] Ready! Loaded {len(ml_service.models)} models.")
    yield
    # Shutdown
    print("[CyberAttack-IDS] Shutting down backend...")


app = FastAPI(
    title="Cyber Attack Detection ML Capstone API",
    description="Defensive Network Intrusion Detection System powered by Optimized ML Models and Ensemble Methods",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Cyber Attack Detection IDS API",
        "version": "1.0.0",
        "models_loaded_count": len(ml_service.models),
        "is_dataset_synthetic": ml_service.dataset_summary.get("dataset_info", {}).get("is_synthetic", True) if ml_service.dataset_summary else True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

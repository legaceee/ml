"""
Aggregated API Router for FastAPI application.
"""

from fastapi import APIRouter

from backend.app.api.dataset import router as dataset_router
from backend.app.api.explain import router as explain_router
from backend.app.api.experiments import router as experiments_router
from backend.app.api.metrics import router as metrics_router
from backend.app.api.models import router as models_router
from backend.app.api.predict import router as predict_router

api_router = APIRouter(prefix="/api")

api_router.include_router(predict_router)
api_router.include_router(models_router)
api_router.include_router(metrics_router)
api_router.include_router(explain_router)
api_router.include_router(experiments_router)
api_router.include_router(dataset_router)

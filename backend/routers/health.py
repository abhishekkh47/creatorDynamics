from fastapi import APIRouter

from predictor import model_store
from schemas import HealthResponse

router = APIRouter(tags=["Infrastructure"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_store.all_loaded else "degraded",
        models=model_store.status,
        models_dir=str(model_store.models_dir),
    )

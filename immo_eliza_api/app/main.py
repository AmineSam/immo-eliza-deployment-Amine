"""
Immo-Eliza FastAPI Application
Production-ready API for Belgian real estate price prediction.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.model_loader import model_loader
from app.routers import predict

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Load models
    logger.info("Starting up Immo-Eliza API...")
    try:
        model_loader.load_all()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Immo-Eliza API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready API for Belgian real estate price prediction using specialized XGBoost models",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, tags=["Prediction"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded models.
    
    Returns model names, performance metrics, and feature list.
    """
    from app.ml.feature_engineering import REDUCED_FEATURES
    
    return {
        "models": {
            "house": {
                "name": "XGBoost House Model",
                "file": "model_xgb_house.pkl",
                "mae_relative_error": settings.house_mae_relative,
                "confidence_interval": f"±{settings.house_mae_relative * 100:.1f}%"
            },
            "apartment": {
                "name": "XGBoost Apartment Model",
                "file": "model_xgb_apartment.pkl",
                "mae_relative_error": settings.apartment_mae_relative,
                "confidence_interval": f"±{settings.apartment_mae_relative * 100:.1f}%"
            }
        },
        "features": {
            "count": len(REDUCED_FEATURES),
            "list": REDUCED_FEATURES
        },
        "preprocessing": {
            "stage3_pipeline": "Fitted preprocessing with target encoding, missingness flags, and log transforms"
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Belgian real estate price prediction API",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

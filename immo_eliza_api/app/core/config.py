"""
Configuration management for Immo-Eliza FastAPI application.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Settings
    app_name: str = "Immo-Eliza Price Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # CORS Settings
    cors_origins: List[str] = ["*"]
    
    # Paths (relative to project root)
    base_dir: Path = Path(__file__).parent.parent.parent
    models_dir: Path = base_dir / "app" / "models"
    data_dir: Path = base_dir / "app" / "data"
    
    # Model files
    model_house_path: Path = models_dir / "model_xgb_house.pkl"
    model_apartment_path: Path = models_dir / "model_xgb_apartment.pkl"
    pipeline_house_path: Path = models_dir / "stage3_pipeline_house.pkl"
    pipeline_apartment_path: Path = models_dir / "stage3_pipeline_apartment.pkl"
    
    # Data files
    lookup_data_path: Path = data_dir / "lookup_data.csv"
    
    # Model Performance Metrics
    house_mae_relative: float = 0.167  # 16.7% relative error
    apartment_mae_relative: float = 0.09  # 9% relative error
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

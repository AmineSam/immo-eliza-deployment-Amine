"""
Model and data loader for Immo-Eliza API.
Loads models, pipelines, and lookup data once at startup.
"""
import pandas as pd
import joblib
import logging
from typing import Dict, Tuple, Any
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class to load and cache models and data."""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self.model_house = None
            self.model_apartment = None
            self.pipeline_house = None
            self.pipeline_apartment = None
            self.lookup_df = None
            self.region_counts = None
    
    def load_all(self) -> None:
        """Load all models, pipelines, and lookup data."""
        if self._models_loaded:
            logger.info("Models already loaded, skipping...")
            return
        
        logger.info("Loading models and data...")
        
        try:
            # Load models
            logger.info(f"Loading house model from {settings.model_house_path}")
            self.model_house = joblib.load(settings.model_house_path)
            
            logger.info(f"Loading apartment model from {settings.model_apartment_path}")
            self.model_apartment = joblib.load(settings.model_apartment_path)
            
            # Load pipelines
            logger.info(f"Loading house pipeline from {settings.pipeline_house_path}")
            self.pipeline_house = joblib.load(settings.pipeline_house_path)
            
            logger.info(f"Loading apartment pipeline from {settings.pipeline_apartment_path}")
            self.pipeline_apartment = joblib.load(settings.pipeline_apartment_path)
            
            # Load lookup data
            logger.info(f"Loading lookup data from {settings.lookup_data_path}")
            df = pd.read_csv(settings.lookup_data_path)
            
            # Create lookup DataFrame
            lookup_cols = [
                "postal_code", "locality", "province", "region", "median_income",
                "province_benchmark_m2", "region_benchmark_m2", "national_benchmark_m2",
                "house_avg_m2_province", "apt_avg_m2_province",
                "house_avg_m2_region", "apt_avg_m2_region"
            ]
            self.lookup_df = df[lookup_cols].drop_duplicates(subset=["postal_code"]).set_index("postal_code")
            
            # Create region counts
            self.region_counts = df["region"].value_counts().to_dict()
            
            ModelLoader._models_loaded = True
            logger.info("All models and data loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_models(self) -> Tuple[Any, Any, Any, Any]:
        """Get loaded models and pipelines."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_all() first.")
        return self.model_house, self.model_apartment, self.pipeline_house, self.pipeline_apartment
    
    def get_lookup_data(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Get lookup DataFrame and region counts."""
        if not self._models_loaded:
            raise RuntimeError("Data not loaded. Call load_all() first.")
        return self.lookup_df, self.region_counts


# Global instance
model_loader = ModelLoader()

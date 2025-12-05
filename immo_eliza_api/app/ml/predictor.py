"""
Main prediction service - replicates Streamlit prediction logic 100%.
"""
import pandas as pd
import logging
from typing import Dict, Tuple

from app.ml.stage3_preprocessing import transform_stage3
from app.ml.feature_engineering import REDUCED_FEATURES, get_metadata, compute_confidence_interval
from app.core.model_loader import model_loader

logger = logging.getLogger(__name__)


def predict_price(property_dict: Dict) -> Tuple[float, float, float, Dict]:
    """
    Predict property price with confidence interval.
    
    Args:
        property_dict: Dictionary with property features including:
            - property_type: "House" or "Apartment"
            - All required features for prediction
            
    Returns:
        Tuple of (predicted_price, ci_low, ci_high, metadata)
    """
    try:
        # Get models and data
        model_house, model_apartment, pipeline_house, pipeline_apartment = model_loader.get_models()
        lookup_df, region_counts = model_loader.get_lookup_data()
        
        # Get metadata for postal code
        postal_code = property_dict["postal_code"]
        metadata = get_metadata(postal_code, lookup_df)
        
        # Enrich input with metadata
        enriched_dict = {**property_dict, **metadata}
        
        # Select model and pipeline based on property type
        if property_dict["property_type"] == "House":
            pipeline = pipeline_house
            model = model_house
        else:
            pipeline = pipeline_apartment
            model = model_apartment
        
        # Create DataFrame
        df_input = pd.DataFrame([enriched_dict])
        
        # Apply Stage 3 preprocessing
        df_s3 = transform_stage3(df_input, pipeline)
        
        # Select features in exact order
        X = df_s3[[f for f in REDUCED_FEATURES if f in df_s3.columns]]
        
        # Predict
        prediction = float(model.predict(X)[0])
        
        # Compute confidence interval
        ci_low, ci_high = compute_confidence_interval(prediction, property_dict["property_type"])
        
        logger.info(f"Prediction: {prediction:.2f}, CI: [{ci_low:.2f}, {ci_high:.2f}]")
        
        return prediction, ci_low, ci_high, metadata
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

"""
Feature engineering utilities for prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from app.core.config import settings


# Exact feature order from Streamlit version
REDUCED_FEATURES = [
    "area",
    "postal_code_te_price",
    "locality_te_price",
    "bathrooms",
    "rooms",
    "primary_energy_consumption",
    "state",
    "province_benchmark_m2",
    "postal_code",
    "region_benchmark_m2",
    "property_subtype_te_price",
    "apt_avg_m2_region",
    "toilets",
    "property_type_te_price",
    "median_income",
    "build_year",
    "house_avg_m2_province",
    "has_garage",
    "apt_avg_m2_province",
    "has_garden",
    "has_terrace",
    "facades_number",
    "has_swimming_pool",
    "house_avg_m2_region",
    "has_equipped_kitchen",
]


def get_metadata(postal_code: int, lookup_df: pd.DataFrame) -> Dict:
    """
    Get metadata for a postal code from lookup DataFrame.
    If postal code not found, use nearest postal code.
    
    Args:
        postal_code: Postal code to lookup
        lookup_df: DataFrame indexed by postal_code
        
    Returns:
        Dictionary with locality, province, region, and benchmark data
    """
    if postal_code in lookup_df.index:
        return lookup_df.loc[postal_code].to_dict()
    
    # Find nearest postal code
    all_pcs = lookup_df.index.values
    nearest = all_pcs[np.abs(all_pcs - postal_code).argmin()]
    return lookup_df.loc[nearest].to_dict()


def compute_confidence_interval(prediction: float, property_type: str) -> Tuple[float, float]:
    """
    Compute confidence interval based on MAE-derived relative error.
    
    Args:
        prediction: Predicted price
        property_type: "House" or "Apartment"
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if property_type.lower().startswith("house"):
        error = settings.house_mae_relative
    else:
        error = settings.apartment_mae_relative
    
    lower = prediction * (1 - error)
    upper = prediction * (1 + error)
    
    return lower, upper

"""
Prediction API router.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
import logging

from app.ml.predictor import predict_price

logger = logging.getLogger(__name__)

router = APIRouter()


# Valid subtypes
HOUSE_SUBTYPES = [
    "residence", "villa", "mixed building", "master house",
    "cottage", "bungalow", "chalet", "mansion"
]

APARTMENT_SUBTYPES = [
    "apartment", "ground floor", "penthouse", "duplex",
    "studio", "loft", "triplex", "student flat", "student housing"
]


class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    
    property_type: Literal["House", "Apartment"] = Field(
        ..., description="Type of property: House or Apartment"
    )
    property_subtype: str = Field(
        ..., description="Specific subtype (e.g., villa, penthouse)"
    )
    postal_code: int = Field(
        ..., ge=1000, le=9999, description="Belgian postal code"
    )
    area: int = Field(
        ..., gt=0, description="Living area in square meters"
    )
    rooms: int = Field(
        ..., ge=0, description="Number of bedrooms"
    )
    bathrooms: int = Field(
        ..., ge=0, description="Number of bathrooms"
    )
    toilets: int = Field(
        ..., ge=0, description="Number of toilets"
    )
    primary_energy_consumption: int = Field(
        ..., ge=0, le=500, description="Primary energy consumption in kWh/m²"
    )
    state: int = Field(
        ..., description="Building state: 0 (needs renovation), 2 (good), 4 (new/renovated)"
    )
    build_year: int = Field(
        ..., ge=1800, le=2030, description="Year the building was constructed"
    )
    facades_number: int = Field(
        ..., ge=0, le=4, description="Number of facades"
    )
    has_garage: bool = Field(default=False, description="Has garage")
    has_garden: bool = Field(default=False, description="Has garden")
    has_terrace: bool = Field(default=False, description="Has terrace")
    has_equipped_kitchen: bool = Field(default=False, description="Has equipped kitchen")
    has_swimming_pool: bool = Field(default=False, description="Has swimming pool")
    
    @validator("property_subtype")
    def validate_subtype(cls, v, values):
        """Validate subtype matches property type."""
        if "property_type" in values:
            prop_type = values["property_type"]
            valid_subtypes = HOUSE_SUBTYPES if prop_type == "House" else APARTMENT_SUBTYPES
            if v.lower() not in valid_subtypes:
                raise ValueError(
                    f"Invalid subtype '{v}' for {prop_type}. "
                    f"Valid subtypes: {', '.join(valid_subtypes)}"
                )
        return v.lower()
    
    @validator("state")
    def validate_state(cls, v):
        """Validate state is one of allowed values."""
        if v not in [0, 2, 4]:
            raise ValueError("State must be 0 (needs renovation), 2 (good), or 4 (new/renovated)")
        return v


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    predicted_price: float = Field(..., description="Predicted property price in EUR")
    confidence_interval_low: float = Field(..., description="Lower bound of confidence interval")
    confidence_interval_high: float = Field(..., description="Upper bound of confidence interval")
    property_type: str = Field(..., description="Type of property")
    postal_code: int = Field(..., description="Postal code")
    locality: str = Field(..., description="Locality name")
    province: str = Field(..., description="Province name")
    region: str = Field(..., description="Region name")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict property price based on input features.
    
    Returns predicted price with confidence interval and location metadata.
    """
    try:
        # Convert request to dictionary
        property_dict = {
            "property_type": request.property_type,
            "property_subtype": request.property_subtype,
            "postal_code": request.postal_code,
            "area": request.area,
            "rooms": request.rooms,
            "bathrooms": request.bathrooms,
            "toilets": request.toilets,
            "primary_energy_consumption": request.primary_energy_consumption,
            "state": request.state,
            "build_year": request.build_year,
            "facades_number": request.facades_number,
            "has_garage": 1 if request.has_garage else 0,
            "has_garden": 1 if request.has_garden else 0,
            "has_terrace": 1 if request.has_terrace else 0,
            "has_equipped_kitchen": 2 if request.has_equipped_kitchen else 0,
            "has_swimming_pool": 1 if request.has_swimming_pool else 0,
        }
        
        # Get prediction
        prediction, ci_low, ci_high, metadata = predict_price(property_dict)
        
        # Build response
        response = PredictionResponse(
            predicted_price=prediction,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            property_type=request.property_type,
            postal_code=request.postal_code,
            locality=metadata.get("locality", ""),
            province=metadata.get("province", ""),
            region=metadata.get("region", "")
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

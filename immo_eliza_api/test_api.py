"""
Simple test script to verify FastAPI endpoints.
Run this after starting the server with: uvicorn app.main:app --reload
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✅ Health check passed!")


def test_model_info():
    """Test model info endpoint."""
    print("\n=== Testing /model-info ===")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Models: {list(data['models'].keys())}")
    print(f"Feature count: {data['features']['count']}")
    assert response.status_code == 200
    assert "models" in data
    assert "house" in data["models"]
    assert "apartment" in data["models"]
    print("✅ Model info check passed!")


def test_predict_house():
    """Test prediction for a house."""
    print("\n=== Testing /predict (House) ===")
    payload = {
        "property_type": "House",
        "property_subtype": "villa",
        "postal_code": 1000,
        "area": 150,
        "rooms": 3,
        "bathrooms": 2,
        "toilets": 2,
        "primary_energy_consumption": 200,
        "state": 2,
        "build_year": 2000,
        "facades_number": 2,
        "has_garage": True,
        "has_garden": True,
        "has_terrace": False,
        "has_equipped_kitchen": True,
        "has_swimming_pool": False
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Price: €{result['predicted_price']:,.0f}")
        print(f"Confidence Interval: €{result['confidence_interval_low']:,.0f} - €{result['confidence_interval_high']:,.0f}")
        print(f"Location: {result['locality']}, {result['province']}, {result['region']}")
        assert "predicted_price" in result
        assert result["predicted_price"] > 0
        print("✅ House prediction passed!")
    else:
        print(f"Error: {response.text}")
        raise AssertionError("Prediction failed")


def test_predict_apartment():
    """Test prediction for an apartment."""
    print("\n=== Testing /predict (Apartment) ===")
    payload = {
        "property_type": "Apartment",
        "property_subtype": "apartment",
        "postal_code": 1050,
        "area": 85,
        "rooms": 2,
        "bathrooms": 1,
        "toilets": 1,
        "primary_energy_consumption": 180,
        "state": 2,
        "build_year": 2010,
        "facades_number": 1,
        "has_garage": False,
        "has_garden": False,
        "has_terrace": True,
        "has_equipped_kitchen": True,
        "has_swimming_pool": False
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Price: €{result['predicted_price']:,.0f}")
        print(f"Confidence Interval: €{result['confidence_interval_low']:,.0f} - €{result['confidence_interval_high']:,.0f}")
        print(f"Location: {result['locality']}, {result['province']}, {result['region']}")
        assert "predicted_price" in result
        assert result["predicted_price"] > 0
        print("✅ Apartment prediction passed!")
    else:
        print(f"Error: {response.text}")
        raise AssertionError("Prediction failed")


def test_invalid_input():
    """Test validation with invalid input."""
    print("\n=== Testing validation (invalid postal code) ===")
    payload = {
        "property_type": "House",
        "property_subtype": "villa",
        "postal_code": 99999,  # Invalid postal code
        "area": 150,
        "rooms": 3,
        "bathrooms": 2,
        "toilets": 2,
        "primary_energy_consumption": 200,
        "state": 2,
        "build_year": 2000,
        "facades_number": 2,
        "has_garage": True,
        "has_garden": True,
        "has_terrace": False,
        "has_equipped_kitchen": True,
        "has_swimming_pool": False
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 422  # Validation error
    print("✅ Validation check passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI Endpoint Testing")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  uvicorn app.main:app --reload")
    print("=" * 60)
    
    try:
        test_health()
        test_model_info()
        test_predict_house()
        test_predict_apartment()
        test_invalid_input()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server.")
        print("Make sure the server is running with:")
        print("  uvicorn app.main:app --reload")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

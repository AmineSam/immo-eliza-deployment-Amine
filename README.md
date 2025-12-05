# 🏠 Immo-Eliza FastAPI Backend

Production-ready REST API for Belgian real estate price prediction using specialized XGBoost models.

## 📋 Overview

This is a complete FastAPI backend that provides price predictions for Belgian properties (houses and apartments) via REST API endpoints. It uses the same ML logic as the Streamlit version, with 100% preservation of preprocessing, feature engineering, and model inference.

### Key Features

- **Dual-model architecture**: Separate XGBoost models for houses and apartments
- **REST API**: Clean JSON endpoints for integration
- **Production-ready**: Docker containerization, health checks, logging
- **Input validation**: Pydantic models with comprehensive validation
- **CORS enabled**: Ready for frontend integration
- **Render deployment**: One-click deployment configuration

---

## 🚀 API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### `GET /model-info`
Get information about loaded models and features.

**Response:**
```json
{
  "models": {
    "house": {
      "name": "XGBoost House Model",
      "file": "model_xgb_house.pkl",
      "mae_relative_error": 0.167,
      "confidence_interval": "±16.7%"
    },
    "apartment": {
      "name": "XGBoost Apartment Model",
      "file": "model_xgb_apartment.pkl",
      "mae_relative_error": 0.09,
      "confidence_interval": "±9.0%"
    }
  },
  "features": {
    "count": 25,
    "list": ["area", "postal_code_te_price", ...]
  }
}
```

### `POST /predict`
Predict property price.

**Request Body:**
```json
{
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
  "has_garage": true,
  "has_garden": true,
  "has_terrace": false,
  "has_equipped_kitchen": true,
  "has_swimming_pool": false
}
```

**Response:**
```json
{
  "predicted_price": 450000.0,
  "confidence_interval_low": 374850.0,
  "confidence_interval_high": 525150.0,
  "property_type": "House",
  "postal_code": 1000,
  "locality": "Brussels",
  "province": "Brussels Hoofdstedelijk Gewest",
  "region": "Brussels"
}
```

---

## 🛠️ Local Development

### Prerequisites

- Python 3.11+
- pip

### Setup

1. **Navigate to project directory**
   ```bash
   cd immo_eliza_api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the API**
   - API: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

---

## 🐳 Docker Usage

### Build Image

```bash
cd immo_eliza_api
docker build -t immo-eliza-api .
```

### Run Container

```bash
docker run -p 8000:8000 immo-eliza-api
```

### Access API

Open `http://localhost:8000/docs` in your browser.

---

## ☁️ Render Deployment

### Option 1: Using render.yaml (Recommended)

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New" → "Blueprint"
4. Connect your repository
5. Render will automatically detect `render.yaml` and deploy

### Option 2: Manual Setup

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Web Service"
3. Connect your repository
4. Configure:
   - **Name**: immo-eliza-api
   - **Environment**: Docker
   - **Dockerfile Path**: `./immo_eliza_api/Dockerfile`
   - **Docker Context**: `./immo_eliza_api`
5. Add environment variables:
   - `PYTHONHASHSEED=0`
   - `PORT=8000`
6. Click "Create Web Service"

### Post-Deployment

- Your API will be available at: `https://your-app-name.onrender.com`
- Health check: `https://your-app-name.onrender.com/health`
- Docs: `https://your-app-name.onrender.com/docs`

---

## 📝 API Usage Examples

### Using curl

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model Info
```bash
curl http://localhost:8000/model-info
```

#### Predict Price
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "has_garage": true,
    "has_garden": true,
    "has_terrace": false,
    "has_equipped_kitchen": true,
    "has_swimming_pool": false
  }'
```

### Using Python

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get model info
response = requests.get(f"{BASE_URL}/model-info")
print(response.json())

# Predict price
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
result = response.json()

print(f"Predicted Price: €{result['predicted_price']:,.0f}")
print(f"Confidence Interval: €{result['confidence_interval_low']:,.0f} - €{result['confidence_interval_high']:,.0f}")
print(f"Location: {result['locality']}, {result['province']}")
```

### Using JavaScript/Fetch

```javascript
// Predict price
const payload = {
  property_type: "House",
  property_subtype: "villa",
  postal_code: 2000,
  area: 200,
  rooms: 4,
  bathrooms: 2,
  toilets: 3,
  primary_energy_consumption: 150,
  state: 4,
  build_year: 2020,
  facades_number: 3,
  has_garage: true,
  has_garden: true,
  has_terrace: true,
  has_equipped_kitchen: true,
  has_swimming_pool: true
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => {
    console.log('Predicted Price:', data.predicted_price);
    console.log('Confidence Interval:', data.confidence_interval_low, '-', data.confidence_interval_high);
  });
```

---

## 📁 Project Structure

```
immo_eliza_api/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   └── model_loader.py        # Model loading singleton
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── stage3_preprocessing.py  # Stage 3 preprocessing (exact copy)
│   │   ├── feature_engineering.py   # Feature utilities
│   │   └── predictor.py             # Main prediction service
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py             # Prediction endpoints
│   │
│   ├── models/                    # Trained models (copied from parent)
│   │   ├── model_xgb_house.pkl
│   │   ├── model_xgb_apartment.pkl
│   │   ├── stage3_pipeline_house.pkl
│   │   └── stage3_pipeline_apartment.pkl
│   │
│   └── data/                      # Lookup data
│       └── lookup_data.csv
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── render.yaml                    # Render deployment config
└── README.md                      # This file
```

---

## 🔧 Configuration

### Environment Variables

You can configure the application using environment variables or a `.env` file:

```env
# API Settings
DEBUG=false
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=["*"]
```

---

## 🧪 Testing

### Manual Testing

Use the interactive API documentation at `/docs` to test all endpoints.

### Automated Testing

Create a test file `test_api.py`:

```python
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = requests.get(f"{BASE_URL}/model-info")
    assert response.status_code == 200
    assert "models" in response.json()

def test_predict():
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
    assert response.status_code == 200
    assert "predicted_price" in response.json()

if __name__ == "__main__":
    test_health()
    test_model_info()
    test_predict()
    print("All tests passed!")
```

Run with:
```bash
python test_api.py
```

---

## 📊 Model Information

### House Model
- **Type**: XGBoost Regressor
- **MAE Relative Error**: 16.7%
- **Confidence Interval**: ±16.7%
- **Trained on**: Belgian house data (villas, residences, mixed buildings, etc.)

### Apartment Model
- **Type**: XGBoost Regressor
- **MAE Relative Error**: 9%
- **Confidence Interval**: ±9%
- **Trained on**: Belgian apartment data (flats, studios, penthouses, etc.)

### Features (25 total)
The models use 25 engineered features including:
- Property characteristics (area, rooms, bathrooms, etc.)
- Target-encoded categorical features
- Geographic benchmarks (province, region, national)
- Amenities (garage, garden, terrace, pool, kitchen)

---

## 🤝 Contributing

This is a production deployment of the Immo-Eliza ML models. For model improvements or feature requests, please refer to the main training repository.

---

## 📄 License

MIT License - see parent repository for details.

---

## 👤 Author

**Amine Sam**
- GitHub: [@AmineSam](https://github.com/AmineSam)
- Main Project: [immo-eliza-deployment-Amine](https://github.com/AmineSam/immo-eliza-deployment-Amine)

---

**Built with ❤️ for the Belgian real estate market**

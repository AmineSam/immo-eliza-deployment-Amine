# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
cd immo_eliza_api
pip install -r requirements.txt
```

### 2. Start the Server

```bash
uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 3. Test the API

Open your browser and go to:
- **Interactive Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000

Or use curl:
```bash
curl http://localhost:8000/health
```

## 📝 Make Your First Prediction

### Using the Interactive Docs (Easiest)

1. Go to http://localhost:8000/docs
2. Click on `POST /predict`
3. Click "Try it out"
4. Use this example:

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

5. Click "Execute"
6. See the predicted price!

### Using curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "has_garage": false,
    "has_garden": false,
    "has_terrace": true,
    "has_equipped_kitchen": true,
    "has_swimming_pool": false
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "property_type": "House",
        "property_subtype": "villa",
        "postal_code": 2000,
        "area": 200,
        "rooms": 4,
        "bathrooms": 2,
        "toilets": 3,
        "primary_energy_consumption": 150,
        "state": 4,
        "build_year": 2020,
        "facades_number": 3,
        "has_garage": True,
        "has_garden": True,
        "has_terrace": True,
        "has_equipped_kitchen": True,
        "has_swimming_pool": True
    }
)

print(response.json())
```

## 🐳 Using Docker

```bash
# Build
docker build -t immo-eliza-api .

# Run
docker run -p 8000:8000 immo-eliza-api

# Access
curl http://localhost:8000/health
```

## 🧪 Run Tests

```bash
# Start server in one terminal
uvicorn app.main:app --reload

# Run tests in another terminal
python test_api.py
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the interactive docs at http://localhost:8000/docs
- Deploy to Render using [render.yaml](render.yaml)

## ❓ Troubleshooting

### Port already in use
```bash
# Use a different port
uvicorn app.main:app --reload --port 8001
```

### Module not found
```bash
# Make sure you're in the right directory
cd immo_eliza_api

# Reinstall dependencies
pip install -r requirements.txt
```

### Models not loading
Check that model files exist in `app/models/`:
- model_xgb_house.pkl
- model_xgb_apartment.pkl
- stage3_pipeline_house.pkl
- stage3_pipeline_apartment.pkl

And data file in `app/data/`:
- lookup_data.csv

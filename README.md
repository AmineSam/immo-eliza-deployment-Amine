# 🏠 Immo-Eliza Real Estate Price Estimator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://www.immo-eliza.be/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Demo:** [https://www.immo-eliza.be/](https://www.immo-eliza.be/)

A production-ready machine learning web application for estimating Belgian real estate prices. Built with Streamlit and powered by specialized XGBoost models trained separately for houses and apartments, this tool provides accurate price predictions with confidence intervals, location insights, and downloadable PDF reports.

---

## 📋 Overview

Immo-Eliza is a comprehensive property valuation tool designed specifically for the Belgian real estate market. The application leverages advanced machine learning techniques to provide accurate price estimates based on property characteristics, location data, and market benchmarks.

The project features:
- **Dual-model architecture**: Separate models optimized for houses and apartments
- **Custom domain**: Deployed at [immo-eliza.be](https://www.immo-eliza.be/)
- **Professional UI**: Clean, modern interface with light/dark mode support
- **Detailed reporting**: Generate PDF reports with predictions and insights
- **Market intelligence**: Price benchmarks and location-based analytics

---

## ✨ Features

### 🎯 Core Functionality
- **Specialized ML Models**: Two separate XGBoost models trained specifically for:
  - Houses (villas, residences, mixed buildings, etc.)
  - Apartments (studios, flats, penthouses, etc.)
- **Accurate Predictions**: Price estimates with confidence intervals based on:
  - Property type and subtype
  - Location (postal code with automatic locality detection)
  - Physical characteristics (area, rooms, bathrooms, facades)
  - Building condition and year
  - Amenities (garden, terrace, pool, parking)

### 📊 Market Insights
- **Price Benchmarks**: Compare estimated price per m² against:
  - National averages
  - Provincial averages
  - Regional averages
- **Location Intelligence**: Automatic metadata enrichment including:
  - Province and region detection
  - Local pricing trends

### 📄 PDF Report Generation
Generate professional PDF reports containing:
- Estimated property price with confidence interval
- All input property details
- Model used (House or Apartment)
- Price per m² analysis
- Location and market context

### 🎨 User Experience
- **Light/Dark Mode Toggle**: Seamless theme switching for user preference
- **Responsive Design**: Clean, modern interface optimized for all screen sizes
- **Input Validation**: Real-time feedback and error handling
- **Visual Analytics**: Interactive charts and visualizations

---

## 🛠️ Tech Stack

### Machine Learning
- **XGBoost**: Gradient boosting framework for regression models
- **scikit-learn**: Data preprocessing and pipeline management
- **NumPy & Pandas**: Data manipulation and numerical computing

### Web Application
- **Streamlit**: Interactive web application framework
- **Altair**: Declarative statistical visualization library
- **FPDF2**: PDF generation with Unicode support

### Data Processing
- **Joblib**: Model serialization and loading
- **Custom Pipelines**: Multi-stage preprocessing and feature engineering

### Deployment
- **Streamlit Cloud**: Hosting platform
- **Custom Domain**: immo-eliza.be with DNS configuration

---

## 📁 Project Structure

```
immo-eliza-deployment-Amine/
│
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── fonts/
│       └── DejaVuSans.ttf         # Unicode font for PDF generation
│
├── models/
│   ├── model_xgb_house.pkl        # XGBoost model for houses
│   ├── model_xgb_apartment.pkl    # XGBoost model for apartments
│   ├── stage3_pipeline_house.pkl  # Preprocessing pipeline for houses
│   └── stage3_pipeline_apartment.pkl  # Preprocessing pipeline for apartments
│
├── utils/
│   ├── predictor.py               # Prediction logic
│   ├── stage3_utils.py            # Feature engineering utilities
│   ├── ml_utils.py                # Machine learning helpers
│   ├── house_optimization_gpu.py  # House model training script
│   └── apartment_optimization_gpu.py  # Apartment model training script
│
├── config/
│   └── [Configuration files]
│
├── data/
│   └── [Training and reference data]
│
├── pipelines/
│   └── [Data preprocessing pipelines]
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AmineSam/immo-eliza-deployment-Amine.git
   cd immo-eliza-deployment-Amine
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

### Using the Application

1. **Select Property Type**: Choose between House or Apartment
2. **Choose Subtype**: Select specific property category (villa, penthouse, etc.)
3. **Enter Location**: Provide postal code (locality auto-detected)
4. **Specify Details**:
   - Build year and condition
   - Living area in m²
   - Number of bedrooms, bathrooms, toilets, facades
5. **Add Amenities**: Check available features (garden, terrace, pool, parking)
6. **Get Estimate**: Click "Estimate Price" to generate prediction
7. **Download Report**: Generate and download PDF report with full details

---

## 🌐 Deployment Notes

### Streamlit Cloud Deployment

The application is deployed on Streamlit Cloud with the following configuration:

- **Main file**: `app/streamlit_app.py`
- **Python version**: 3.9+
- **Custom domain**: immo-eliza.be (configured via DNS CNAME)

### Domain Configuration

1. **Domain purchased**: immo-eliza.be
2. **DNS Setup**: CNAME record pointing to Streamlit Cloud
3. **SSL/TLS**: Automatically managed by Cloudflare
4. **Redirect**: [www.immo-eliza.be](https://www.immo-eliza.be/) → Streamlit app using Cloudflare

### Environment Considerations

- Models are loaded once at startup for performance
- Session state manages theme preferences
- PDF generation uses bundled DejaVuSans.ttf font for Unicode support
- No external API dependencies for core functionality

---


## 📊 Model Performance

The specialized models were trained on Belgian real estate data with the following characteristics:

- **House Model**: Optimized for single-family homes, villas, and mixed buildings
- **Apartment Model**: Optimized for apartments, studios, penthouses, and flats
- **Training Approach**: GPU-accelerated hyperparameter tuning using Optuna
- **Validation**: Cross-validation with confidence interval estimation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Amine Sam**

- GitHub: [@AmineSam](https://github.com/AmineSam)
- Project: [immo-eliza-deployment-Amine](https://github.com/AmineSam/immo-eliza-deployment-Amine)
- This project was developed as part of the AI & Data Science Bootcamp at BeCode.org 

---

## 🙏 Acknowledgments

- Belgian real estate data sources
- Streamlit community for excellent documentation
- XGBoost developers for the powerful ML framework
- Open-source contributors

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/AmineSam/immo-eliza-deployment-Amine/issues)
- Visit the live app at [immo-eliza.be](https://www.immo-eliza.be/)

---

**Built with ❤️ for the Belgian real estate market**
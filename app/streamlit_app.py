import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import altair as alt
import xgboost as xgb
import io

# PDF generation
# PDF generation
from fpdf import FPDF

# Add root to sys.path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.stage3_utils import transform_stage3
except ImportError:
    st.error("Could not import utils.stage3_utils. Please run app from the repo root.")
    st.stop()

# =========================================================
# CONFIGURATION & STYLING
# =========================================================

st.set_page_config(
    page_title="Belgian Property Valuation Tool",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

# Theme-specific CSS
def get_theme_css(theme):
    if theme == 'dark':
        return """
<style>
    .stApp {
        background-color: #1a1a2e;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #f0f0f0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 { font-size: 2.5rem; }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #3a3a52;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    .section-icon { color: #60a5fa; margin-right: 0.5rem; }

    .result-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4),
                    0 4px 6px -2px rgba(0,0,0,0.2);
        border: 1px solid #4a5568;
    }
    .result-title {
        color: #60a5fa;
        font-size: 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    .result-price {
        color: #93c5fd;
        font-size: 4rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    .result-subline {
        color: #cbd5e0;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .result-disclaimer {
        color: #a0aec0;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 1.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem 3rem;
        border: none;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(59,130,246,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(59,130,246,0.4);
    }
    
    .sidebar-box {
        background-color: #2d3748;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #4a5568;
    }
    
    /* Streamlit specific dark mode overrides */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e0e0e0 !important;
    }
    
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #16213e;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #60a5fa;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d3748;
        color: #e0e0e0;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #10b981;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(16,185,129,0.3);
    }
    
    .stDownloadButton>button:hover {
        background-color: #059669;
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(16,185,129,0.4);
    }
</style>
"""
    else:  # light theme
        return """
<style>
    .stApp {
        background-color: #ffffff;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 { font-size: 2.5rem; }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    .section-icon { color: #3182ce; margin-right: 0.5rem; }

    .result-card {
        background-color: #ebf8ff;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1),
                    0 4px 6px -2px rgba(0,0,0,0.05);
        border: 1px solid #bee3f8;
    }
    .result-title {
        color: #2b6cb0;
        font-size: 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    .result-price {
        color: #2c5282;
        font-size: 4rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    .result-subline {
        color: #4a5568;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .result-disclaimer {
        color: #718096;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 1.5rem;
    }
    
    .stButton>button {
        background-color: #3182ce;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem 3rem;
        border: none;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(50,130,206,0.2);
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(50,130,206,0.3);
    }
    
    .sidebar-box {
        background-color: #e6f0fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #10b981;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(16,185,129,0.2);
    }
    
    .stDownloadButton>button:hover {
        background-color: #059669;
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(16,185,129,0.3);
    }
</style>
"""

# Apply theme CSS
st.markdown(get_theme_css(st.session_state['theme']), unsafe_allow_html=True)

# =========================================================
# DATA & MODEL LOADING
# =========================================================

@st.cache_resource
def load_resources():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(base, "data", "pre_processed", "pre_processed_data_for_kaggle.csv")
    model_house_path = os.path.join(base, "models", "model_xgb_house.pkl")
    model_apt_path = os.path.join(base, "models", "model_xgb_apartment.pkl")
    pipeline_house_path = os.path.join(base, "models", "stage3_pipeline_house.pkl")
    pipeline_apt_path = os.path.join(base, "models", "stage3_pipeline_apartment.pkl")

    try:
        df = pd.read_csv(data_path)
    except Exception:
        st.error("Could not load pre_processed_data_for_kaggle.csv")
        st.stop()

    region_counts = df["region"].value_counts().to_dict()

    lookup_cols = [
        "postal_code", "locality", "province", "region", "median_income",
        "province_benchmark_m2", "region_benchmark_m2", "national_benchmark_m2",
        "house_avg_m2_province", "apt_avg_m2_province",
        "house_avg_m2_region", "apt_avg_m2_region"
    ]
    lookup_df = df[lookup_cols].drop_duplicates(subset=["postal_code"]).set_index("postal_code")

    model_house = joblib.load(model_house_path)
    model_apt = joblib.load(model_apt_path)
    stage3_house = joblib.load(pipeline_house_path)
    stage3_apt = joblib.load(pipeline_apt_path)

    return lookup_df, region_counts, model_house, model_apt, stage3_house, stage3_apt


lookup_df, region_counts, model_house, model_apt, stage3_house, stage3_apt = load_resources()

# =========================================================
# HELPERS
# =========================================================

def get_metadata(pc, lookup_df):
    if pc in lookup_df.index:
        return lookup_df.loc[pc].to_dict()
    all_pcs = lookup_df.index.values
    nearest = all_pcs[np.abs(all_pcs - pc).argmin()]
    return lookup_df.loc[nearest].to_dict()


def compute_model_confidence(pred, model_label):
    err = 0.167 if model_label.lower().startswith("house") else 0.09
    return pred * (1 - err), pred * (1 + err)


def generate_pdf_report(prediction, ci_low, ci_high, data, prop_type, subtype, state_label, similar_count):
    class PDF(FPDF):
        def header(self):
            # Title
            self.set_font("DejaVu", "B", 16)
            self.cell(0, 10, "ImmoEliza Property Valuation Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)
            
            # Subtitle
            self.set_font("DejaVu", "", 12)
            self.cell(0, 10, "AI Real Estate Valuator for Belgian Properties", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(10)

    pdf = PDF()
    
    # Load Unicode font
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)
    
    pdf.add_page()
    
    # 1. Estimated Value Section
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, f"Estimated Value: € {prediction:,.0f}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, f"Confidence Range: € {ci_low:,.0f} - € {ci_high:,.0f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Based on {similar_count:,} similar properties.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # 2. Property Summary
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Property Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_font("DejaVu", "", 12)
    summary_lines = [
        f"Type: {prop_type} ({subtype})",
        f"Postal Code: {data['postal_code']}",
        f"Locality: {data['locality']}",
        f"Living Area: {data['area']} m²",
        f"Bedrooms: {data['rooms']}",
        f"Bathrooms: {data['bathrooms']}",
        f"Toilets: {data['toilets']}",
        f"Facades: {data['facades_number']}",
        f"Build Year: {data['build_year']}",
        f"Building State: {state_label}",
        f"Energy Consumption: {data['primary_energy_consumption']} kWh/m²"
    ]
    
    for line in summary_lines:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # 3. Amenities
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Amenities", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_font("DejaVu", "", 12)
    amenity_lines = [
        f"Garage: {'Yes' if data['has_garage'] else 'No'}",
        f"Garden: {'Yes' if data['has_garden'] else 'No'}",
        f"Terrace: {'Yes' if data['has_terrace'] else 'No'}",
        f"Equipped Kitchen: {'Yes' if data['has_equipped_kitchen'] else 'No'}",
        f"Swimming Pool: {'Yes' if data['has_swimming_pool'] else 'No'}"
    ]
    
    for line in amenity_lines:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # 4. Location Insights
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Location Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_font("DejaVu", "", 12)
    loc_lines = [
        f"Province: {data['province']}",
        f"Region: {data['region']}",
        f"Median Income: € {data['median_income']:,.0f}",
        f"Province Benchmark: € {data['province_benchmark_m2']:,.0f}/m²",
        f"Region Benchmark: € {data['region_benchmark_m2']:,.0f}/m²",
        f"National Benchmark: € {data['national_benchmark_m2']:,.0f}/m²"
    ]
    
    for line in loc_lines:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(15)
    
    # Disclaimer
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 5, "Disclaimer: Estimate may vary depending on market conditions.")

    return bytes(pdf.output(dest="S"))

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/AmineSam/immo-eliza-deployment-Amine/main/images/%E2%80%94Pngtree%E2%80%94financial%20investment%20real%20estate%20house_7128805.png",
        width=120,
    )
    st.title("ImmoEliza")

    st.markdown(
        """
    <div class="sidebar-box">
        This valuation tool analyzes thousands of real estate transactions in Belgium to estimate property prices.
        <br><br>
        <b>Models:</b><br>
        🏠 <b>House Model</b> – Detached, semi-detached, terraced.<br>
        🏢 <b>Apartment Model</b> – Flats, duplexes, studios.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("v1.0.0 | Production Build")

# =========================================================
# MAIN CONTENT
# =========================================================

# Title with theme toggle
title_col, toggle_col = st.columns([4, 1])

with title_col:
    st.title("Belgian Property Valuation Tool")

with toggle_col:
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    current_theme = st.session_state['theme']
    is_dark = current_theme == 'dark'
    
    if st.toggle("🌙 Dark Mode" if not is_dark else "☀️ Light Mode", value=is_dark, key="theme_toggle"):
        if current_theme == 'light':
            st.session_state['theme'] = 'dark'
            st.rerun()
    else:
        if current_theme == 'dark':
            st.session_state['theme'] = 'light'
            st.rerun()

# ------------------------
# SECTION 1: Property Details
# ------------------------
st.markdown(
    '<div class="section-header"><span class="section-icon">▸</span> Property Details</div>',
    unsafe_allow_html=True,
)
col1_1, col1_2 = st.columns(2)

with col1_1:
    prop_type = st.selectbox(
        "Property Type",
        ["House", "Apartment"],
        index=None,
        placeholder="Select type...",
    )

    HOUSE_SUBTYPES = [
        "residence",
        "villa",
        "mixed building",
        "master house",
        "cottage",
        "bungalow",
        "chalet",
        "mansion",
    ]
    APARTMENT_SUBTYPES = [
        "apartment",
        "ground floor",
        "penthouse",
        "duplex",
        "studio",
        "loft",
        "triplex",
        "student flat",
        "student housing",
    ]

    if prop_type is not None:
        subtype_list = HOUSE_SUBTYPES if prop_type == "House" else APARTMENT_SUBTYPES
        subtype_ui = st.selectbox(
            "Property Subtype",
            [s.title() for s in subtype_list],
            index=None,
            placeholder="Select subtype...",
        )
        prop_subtype = subtype_ui.lower() if subtype_ui else None
    else:
        st.selectbox(
            "Property Subtype",
            ["Select property type first"],
            index=0,
            disabled=True,
        )
        prop_subtype = None

    all_postal_codes = sorted(lookup_df.index.unique().tolist())
    postal_code = st.selectbox(
        "Postal Code",
        all_postal_codes,
        index=None,
        placeholder="Select postal code...",
    )

with col1_2:
    build_year = st.number_input(
        "Build Year",
        min_value=1800,
        max_value=2030,
        value=None,
        placeholder="e.g. 1990",
    )

    state_ui = st.selectbox(
        "Building State",
        ["New / Recently Renovated", "Good Condition", "Needs Renovation"],
        index=None,
        placeholder="Select condition...",
    )

    state_map = {
        "New / Recently Renovated": 4,
        "Good Condition": 2,
        "Needs Renovation": 0,
    }
    state_val = state_map.get(state_ui, 2)

# ------------------------
# SECTION 2: Size & Energy
# ------------------------
st.markdown(
    '<div class="section-header"><span class="section-icon">▸</span> Size & Energy</div>',
    unsafe_allow_html=True,
)
col2_1, col2_2 = st.columns(2)

with col2_1:
    area = st.slider("Living Area (m²)", 15, 500, 120)
with col2_2:
    energy = st.slider("Primary Energy Consumption (kWh/m²)", 0, 500, 250)

# ------------------------
# SECTION 3: Rooms & Layout
# ------------------------
st.markdown(
    '<div class="section-header"><span class="section-icon">▸</span> Rooms & Layout</div>',
    unsafe_allow_html=True,
)
col3_1, col3_2, col3_3, col3_4 = st.columns(4)

with col3_1:
    rooms = st.number_input("Bedrooms", 0, 10, 3)
with col3_2:
    bathrooms = st.number_input("Bathrooms", 0, 5, 1)
with col3_3:
    toilets = st.number_input("Toilets", 0, 5, 2)
with col3_4:
    facades = st.number_input("Facades", 0, 4, 2)

# ------------------------
# SECTION 4: Amenities
# ------------------------
st.markdown(
    '<div class="section-header"><span class="section-icon">▸</span> Amenities</div>',
    unsafe_allow_html=True,
)
col4_1, col4_2 = st.columns(2)

with col4_1:
    has_garage = st.toggle("Garage")
    has_garden = st.toggle("Garden")
    has_terrace = st.toggle("Terrace")

with col4_2:
    has_kitchen = st.toggle("Equipped Kitchen")
    has_pool = st.toggle("Swimming Pool")

st.markdown("---")

# Centered Estimate button
button_cols = st.columns([1, 2, 1])
with button_cols[1]:
    submitted = st.button("Estimate Price")

# =========================================================
# PREDICTION LOGIC
# =========================================================

if submitted:
    if not prop_type or not prop_subtype or not postal_code or not state_ui or not build_year:
        st.error(
            "⚠️ Please fill in all required fields (Type, Subtype, Postal Code, Build Year, State)."
        )
    else:
        with st.spinner("Analyzing market data..."):
            metadata = get_metadata(postal_code, lookup_df)
            region_name = metadata.get("region", "Belgium")
            similar_count = region_counts.get(region_name, 1000)

            input_dict = {
                "property_type": prop_type,
                "property_subtype": prop_subtype,
                "postal_code": postal_code,
                "locality": metadata.get("locality", ""),
                "area": area,
                "rooms": rooms,
                "bathrooms": bathrooms,
                "toilets": toilets,
                "primary_energy_consumption": energy,
                "state": state_val,
                "build_year": build_year,
                "facades_number": facades,
                "has_garage": 1 if has_garage else 0,
                "has_garden": 1 if has_garden else 0,
                "has_terrace": 1 if has_terrace else 0,
                "has_equipped_kitchen": 2 if has_kitchen else 0,
                "has_swimming_pool": 1 if has_pool else 0,
                "median_income": metadata.get("median_income", 0),
                "province": metadata.get("province", ""),
                "region": metadata.get("region", ""),
                "province_benchmark_m2": metadata.get("province_benchmark_m2", 0),
                "region_benchmark_m2": metadata.get("region_benchmark_m2", 0),
                "national_benchmark_m2": metadata.get("national_benchmark_m2", 0),
                "house_avg_m2_province": metadata.get("house_avg_m2_province", 0),
                "apt_avg_m2_province": metadata.get("apt_avg_m2_province", 0),
                "house_avg_m2_region": metadata.get("house_avg_m2_region", 0),
                "apt_avg_m2_region": metadata.get("apt_avg_m2_region", 0),
            }

            try:
                if prop_type == "House":
                    pipeline = stage3_house
                    model = model_house
                else:
                    pipeline = stage3_apt
                    model = model_apt

                df_input = pd.DataFrame([input_dict])
                df_s3 = transform_stage3(df_input, pipeline)

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

                X = df_s3[[f for f in REDUCED_FEATURES if f in df_s3.columns]]
                prediction = float(model.predict(X)[0])
                ci_low, ci_high = compute_model_confidence(prediction, prop_type)
                
                # Store in session state
                st.session_state['prediction_result'] = {
                    'prediction': prediction,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'input_dict': input_dict,
                    'similar_count': similar_count,
                    'region_name': region_name,
                    'prop_type': prop_type,
                    'prop_subtype': prop_subtype,
                    'state_ui': state_ui,
                    'area': area
                }

            except Exception as e:
                st.error(f"An error occurred during valuation: {str(e)}")

# Display Results if available in session state
if 'prediction_result' in st.session_state:
    res = st.session_state['prediction_result']
    prediction = res['prediction']
    ci_low = res['ci_low']
    ci_high = res['ci_high']
    input_dict = res['input_dict']
    similar_count = res['similar_count']
    region_name = res['region_name']
    area = res['area']

    # =========================
    # 5. Result Card with CI
    # =========================
    st.markdown(
        f"""
<div class="result-card">
<div class="result-title">Estimated Property Value</div>
<div class="result-price">€ {prediction:,.0f}</div>
<div style="font-size:1.1rem; margin-top:10px; color:#2c5282;">
<b>Confidence Range:</b><br>
€ {ci_low:,.0f} - € {ci_high:,.0f}
</div>
<div class="result-subline">
Based on over <strong>{similar_count:,}</strong> similar properties in your region ({region_name}).
</div>
<div class="result-disclaimer">
Disclaimer: Estimate based on similar properties in your area. Actual market value may vary.
</div>
</div>
""",
        unsafe_allow_html=True
    )

    # =========================
    # 6. PDF Download (full recap + location insights)
    # =========================
    pdf_buffer = generate_pdf_report(
        prediction,
        ci_low,
        ci_high,
        input_dict,
        res['prop_type'],
        res['prop_subtype'],
        res['state_ui'],
        similar_count,
    )

    st.download_button(
        label="Download Valuation PDF",
        data=pdf_buffer,
        file_name="immoeliza_valuation.pdf",
        mime="application/pdf",
    )

    # =========================
    # 7. Charts & Benchmarks
    # =========================
    st.markdown("---")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res2:
        st.markdown("### Price Benchmarks")

        prov_bench_price = (
            input_dict["province_benchmark_m2"] * area
            if input_dict["province_benchmark_m2"] is not None
            else 0
        )
        reg_bench_price = (
            input_dict["region_benchmark_m2"] * area
            if input_dict["region_benchmark_m2"] is not None
            else 0
        )

        chart_data = pd.DataFrame(
            {
                "Category": ["Predicted Price", "Province Avg", "Region Avg"],
                "Price": [prediction, prov_bench_price, reg_bench_price],
            }
        )

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Category", sort=None),
                y="Price",
                tooltip=[
                    "Category",
                    alt.Tooltip("Price", format=",.0f"),
                ],
            )
            .properties(height=300)
        )

        text = chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
        ).encode(text=alt.Text("Price", format=",.0f"))

        st.altair_chart(chart + text, use_container_width=True)

    # =========================
    # 8. Location Insights (UI)
    # =========================
    with st.expander("See Location Insights", expanded=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Province", input_dict["province"])
        m2.metric("Region", input_dict["region"])
        m3.metric("Locality", input_dict["locality"])

        st.markdown("#### Market Benchmarks (€/m²)")
        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Province Avg",
            f"€ {input_dict['province_benchmark_m2']:,.0f}",
        )
        b2.metric(
            "Region Avg",
            f"€ {input_dict['region_benchmark_m2']:,.0f}",
        )
        b3.metric(
            "National Avg",
            f"€ {input_dict['national_benchmark_m2']:,.0f}",
        )

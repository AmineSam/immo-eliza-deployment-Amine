import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import altair as alt
import xgboost as xgb
import base64

# Add the root directory to sys.path to allow imports from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import stage3_utils
try:
    from utils.stage3_utils import transform_stage3
except ImportError:
    try:
        from utils.stage3_utils import transform_stage3
    except ImportError:
        st.error("Could not import utils.stage3_utils. Please ensure the app is run from the project root or app directory.")
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

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #ffffff;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 { font-size: 2.5rem; }
    
    /* Section Headers */
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
    .section-icon {
        color: #3182ce;
        margin-right: 0.5rem;
    }

    /* Result Card */
    .result-card {
        background-color: #ebf8ff; /* Soft light blue */
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
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
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    .result-disclaimer {
        color: #718096;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 1.5rem;
    }
    .result-subline {
        color: #4a5568;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3182ce;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(50, 130, 206, 0.2);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(50, 130, 206, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA & MODELS
# =========================================================

@st.cache_resource
def load_resources():
    """Load models, pipelines, and data once."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    data_path = os.path.join(base_dir, "data", "pre_processed", "pre_processed_data_for_kaggle.csv")
    model_house_path = os.path.join(base_dir, "models", "model_xgb_house.pkl")
    model_apt_path = os.path.join(base_dir, "models", "model_xgb_apartment.pkl")
    pipeline_house_path = os.path.join(base_dir, "models", "stage3_pipeline_house.pkl")
    pipeline_apt_path = os.path.join(base_dir, "models", "stage3_pipeline_apartment.pkl")

    # Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}")
        st.stop()

    # Calculate region counts for "X similar properties"
    region_counts = df['region'].value_counts().to_dict()
    
    # Create Lookup Table
    lookup_cols = [
        "postal_code", "locality", "province", "region", "median_income",
        "province_benchmark_m2", "region_benchmark_m2", "national_benchmark_m2",
        "house_avg_m2_province", "apt_avg_m2_province", 
        "house_avg_m2_region", "apt_avg_m2_region"
    ]
    available_cols = [c for c in lookup_cols if c in df.columns]
    lookup_df = df[available_cols].drop_duplicates(subset=["postal_code"]).set_index("postal_code")
    
    # Load Models
    model_house = joblib.load(model_house_path)
    model_apt = joblib.load(model_apt_path)
    stage3_house = joblib.load(pipeline_house_path)
    stage3_apt = joblib.load(pipeline_apt_path)

    return lookup_df, region_counts, model_house, model_apt, stage3_house, stage3_apt

lookup_df, region_counts, model_house, model_apt, stage3_house, stage3_apt = load_resources()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_metadata(postal_code, lookup_df):
    if postal_code in lookup_df.index:
        return lookup_df.loc[postal_code].to_dict()
    available_pcs = lookup_df.index.values
    idx = (np.abs(available_pcs - postal_code)).argmin()
    nearest_pc = available_pcs[idx]
    return lookup_df.loc[nearest_pc].to_dict()

def compute_model_confidence(pred_price, model_label):
    HOUSE_RELATIVE_ERROR = 0.167
    APT_RELATIVE_ERROR = 0.09
    rel_err = HOUSE_RELATIVE_ERROR if model_label.lower().startswith("house") else APT_RELATIVE_ERROR
    return pred_price * (1 - rel_err), pred_price * (1 + rel_err)

def create_download_link(html_content, filename="valuation_report.html"):
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}" class="stButton" style="text-decoration:none; color:white; background-color:#3182ce; padding:0.5rem 1rem; border-radius:8px;">Download Valuation Report (PDF)</a>'

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    # Logo (House + Upward Arrow SVG)
    # st.markdown("""
    # <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#2c5282" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    #     <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
    #     <polyline points="9 22 9 12 15 12 15 22"></polyline>
    #     <line x1="12" y1="16" x2="12" y2="10"></line>
    #     <polyline points="9 13 12 10 15 13"></polyline>
    # </svg>
    # """, unsafe_allow_html=True)
    "🏠"
    st.title("ImmoEliza")
    st.markdown("### AI Real Estate Valuator")
    
    st.markdown("## About this Tool")
    st.markdown("""
    This valuation tool analyzes thousands of real estate transactions in Belgium to estimate property prices.
    
    **Models:**
    *   🏠 **House Model** – Detached, semi-detached, terraced.
    *   🏢 **Apartment Model** – Flats, duplexes, studios.
    """)
    
    st.markdown("---")
    st.caption("v1.0.0 | Production Build")

# =========================================================
# MAIN CONTENT
# =========================================================

st.title("Belgian Property Valuation Tool")

# Form
with st.form("valuation_form"):
    
    # SECTION 1: Property Details
    st.markdown('<div class="section-header"><span class="section-icon">▸</span> Property Details</div>', unsafe_allow_html=True)
    
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        prop_type = st.selectbox("Property Type", ["House", "Apartment"], index=None, placeholder="Select type...")
        
        # Subtype Logic
        HOUSE_SUBTYPES = ["residence", "villa", "mixed building", "master house", "cottage", "bungalow", "chalet", "mansion"]
        APARTMENT_SUBTYPES = ["apartment", "ground floor", "penthouse", "duplex", "studio", "loft", "triplex", "student flat", "student housing"]
        
        subtypes_list = []
        if prop_type == "House":
            subtypes_list = HOUSE_SUBTYPES
        elif prop_type == "Apartment":
            subtypes_list = APARTMENT_SUBTYPES
            
        subtypes_ui = [s.title() for s in subtypes_list]
        subtype_ui = st.selectbox("Property Subtype", subtypes_ui, index=None, placeholder="Select subtype...")
        prop_subtype = subtype_ui.lower() if subtype_ui else None
        
        all_postal_codes = sorted(lookup_df.index.unique().tolist())
        postal_code = st.selectbox("Postal Code", all_postal_codes, index=None, placeholder="Select postal code...")

    with col1_2:
        build_year = st.number_input("Build Year", min_value=1800, max_value=2030, value=None, placeholder="e.g. 1990")
        
        # Simplified State Logic
        state_ui = st.selectbox("Building State", ["New / Recently Renovated", "Good Condition", "Needs Renovation"], index=None, placeholder="Select condition...")
        
        state_map = {
            "New / Recently Renovated": 4, # AS_NEW
            "Good Condition": 2,           # GOOD
            "Needs Renovation": 0          # TO_RENOVATE
        }
        state_val = state_map.get(state_ui, 2) # Default to 2 if None, but we validate later

    # SECTION 2: Size & Energy
    st.markdown('<div class="section-header"><span class="section-icon">▸</span> Size & Energy</div>', unsafe_allow_html=True)
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        area = st.slider("Living Area (m²)", 15, 500, 120)
    with col2_2:
        energy = st.slider("Primary Energy Consumption (kWh/m²)", 0, 500, 250)

    # SECTION 3: Rooms & Layout
    st.markdown('<div class="section-header"><span class="section-icon">▸</span> Rooms & Layout</div>', unsafe_allow_html=True)
    
    col3_1, col3_2, col3_3, col3_4 = st.columns(4)
    with col3_1:
        rooms = st.number_input("Bedrooms", 0, 10, 3)
    with col3_2:
        bathrooms = st.number_input("Bathrooms", 0, 5, 1)
    with col3_3:
        toilets = st.number_input("Toilets", 0, 5, 2)
    with col3_4:
        facades = st.number_input("Facades", 0, 4, 2)

    # SECTION 4: Amenities
    st.markdown('<div class="section-header"><span class="section-icon">▸</span> Amenities</div>', unsafe_allow_html=True)
    
    col4_1, col4_2 = st.columns(2)
    with col4_1:
        has_garage = st.toggle("Garage")
        has_garden = st.toggle("Garden")
        has_terrace = st.toggle("Terrace")
    with col4_2:
        has_kitchen = st.toggle("Equipped Kitchen")
        has_pool = st.toggle("Swimming Pool")

    st.markdown("---")
    submitted = st.form_submit_button("Estimate Price", use_container_width=True)

# =========================================================
# PREDICTION LOGIC
# =========================================================

if submitted:
    # Validation
    if not prop_type or not prop_subtype or not postal_code or not state_ui or not build_year:
        st.error("⚠️ Please fill in all required fields (Type, Subtype, Postal Code, Build Year, State).")
    else:
        with st.spinner("Analyzing market data..."):
            # 1. Metadata & Region Count
            metadata = get_metadata(postal_code, lookup_df)
            region_name = metadata.get("region", "Belgium")
            similar_count = region_counts.get(region_name, 1000)
            
            # 2. Input Dict
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
                "has_equipped_kitchen": 2 if has_kitchen else 0, # Fully equipped
                "has_swimming_pool": 1 if has_pool else 0,
                
                # Metadata
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
            
            # 3. Predict
            try:
                # Select Model
                if prop_type == "House":
                    pipeline = stage3_house
                    model = model_house
                else:
                    pipeline = stage3_apt
                    model = model_apt
                
                # Transform & Predict
                df_input = pd.DataFrame([input_dict])
                df_s3 = transform_stage3(df_input, pipeline)
                
                REDUCED_FEATURES = [
                    "area", "postal_code_te_price", "locality_te_price", "bathrooms", "rooms",
                    "primary_energy_consumption", "state", "province_benchmark_m2", "postal_code",
                    "region_benchmark_m2", "property_subtype_te_price", "apt_avg_m2_region",
                    "toilets", "property_type_te_price", "median_income", "build_year",
                    "house_avg_m2_province", "has_garage", "apt_avg_m2_province", "has_garden",
                    "has_terrace", "facades_number", "has_swimming_pool", "house_avg_m2_region",
                    "has_equipped_kitchen"
                ]
                
                X = df_s3[[f for f in REDUCED_FEATURES if f in df_s3.columns]]
                prediction = float(model.predict(X)[0])
                
                # Confidence Interval
                ci_low, ci_high = compute_model_confidence(prediction, prop_type)
                
                # 4. Display Result Card
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Estimated Property Value</div>
                    <div class="result-price">€ {prediction:,.0f}</div>
                    <div class="result-subline">This estimate is based on over <strong>{similar_count:,}</strong> similar properties in your region ({region_name}).</div>
                    <div class="result-disclaimer">Disclaimer: Estimate based on similar properties in your area. Actual market value may vary.</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 5. Export Button
                report_html = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Helvetica, Arial, sans-serif; padding: 40px; background-color: #f7fafc; }}
                        .card {{ background: white; padding: 40px; border-radius: 16px; max-width: 600px; margin: 0 auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }}
                        h1 {{ color: #2c5282; }}
                        .price {{ font-size: 48px; color: #2b6cb0; font-weight: bold; margin: 20px 0; }}
                        .details {{ text-align: left; margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; }}
                        .row {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
                    </style>
                </head>
                <body>
                    <div class="card">
                        <h1>Property Valuation Report</h1>
                        <p>ImmoEliza AI Valuation Tool</p>
                        <div class="price">€ {prediction:,.0f}</div>
                        <p>Based on {similar_count:,} similar properties in {region_name}.</p>
                        <div class="details">
                            <h3>Property Details</h3>
                            <div class="row"><span>Type:</span> <strong>{prop_type} ({subtype_ui})</strong></div>
                            <div class="row"><span>Location:</span> <strong>{postal_code}</strong></div>
                            <div class="row"><span>Area:</span> <strong>{area} m²</strong></div>
                            <div class="row"><span>Bedrooms:</span> <strong>{rooms}</strong></div>
                            <div class="row"><span>Energy Score:</span> <strong>{energy} kWh/m²</strong></div>
                            <div class="row"><span>State:</span> <strong>{state_ui}</strong></div>
                        </div>
                    </div>
                </body>
                </html>
                """
                st.markdown(create_download_link(report_html), unsafe_allow_html=True)
                
                # 6. Charts & Insights (Preserved)
                st.markdown("---")
                
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.markdown("### Market Context")
                    st.metric(label="Median Income (Province)", value=f"€ {input_dict['median_income']:,.0f}")
                    st.caption(f"Confidence Range: € {ci_low:,.0f} - € {ci_high:,.0f}")

                with col_res2:
                    st.markdown("### Price Benchmarks")
                    prov_bench_price = input_dict['province_benchmark_m2'] * area
                    reg_bench_price = input_dict['region_benchmark_m2'] * area
                    
                    chart_data = pd.DataFrame({
                        'Category': ['Predicted Price', 'Province Avg', 'Region Avg'],
                        'Price': [prediction, prov_bench_price, reg_bench_price],
                        'Color': ['#2ecc71', '#3498db', '#95a5a6']
                    })
                    
                    c = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('Category', sort=None),
                        y='Price',
                        color=alt.Color('Color', scale=None),
                        tooltip=['Category', alt.Tooltip('Price', format=',.0f')]
                    ).properties(height=300)
                    
                    text = c.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('Price', format=',.0f'))
                    st.altair_chart(c + text, use_container_width=True)

                # Location Insights (Expanded by default)
                with st.expander("See Location Insights", expanded=True):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Province", input_dict['province'])
                    m2.metric("Region", input_dict['region'])
                    m3.metric("Locality", input_dict['locality'])
                    
                    st.markdown("#### Market Benchmarks (€/m²)")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Province Avg", f"€ {input_dict['province_benchmark_m2']:,.0f}")
                    b2.metric("Region Avg", f"€ {input_dict['region_benchmark_m2']:,.0f}")
                    b3.metric("National Avg", f"€ {input_dict['national_benchmark_m2']:,.0f}")
                    
            except Exception as e:
                st.error(f"An error occurred during valuation: {str(e)}")

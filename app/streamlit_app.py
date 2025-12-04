import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import altair as alt
from fpdf import FPDF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from utils.stage3_utils import transform_stage3
except:
    st.error("Error: Could not import Stage 3 pipeline utilities.")
    st.stop()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Belgian Property Valuation Tool",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.stApp { background-color: white; font-family: "Helvetica Neue"; }
.section-header { font-size: 1.1rem; font-weight: 600; border-bottom:1px solid #eee; margin-top:1.4rem; }
.section-icon { color:#3182ce; margin-right:6px; }

.sidebar-box {
    background-color:#e6f0fa;
    padding:16px;
    border-radius:10px;
    margin-bottom:16px;
}

.result-card {
    background:#ebf8ff;
    border:1px solid #c3ddf6;
    border-radius:14px;
    padding:2rem;
    max-width:650px;
    margin:20px auto;
    text-align:center;
}
.result-title { color:#2b6cb0; font-size:1.1rem; margin-bottom:10px; }
.result-price { color:#2c5282; font-size:3rem; font-weight:800; }
.result-subline { color:#4a5568; margin-top:8px; }
.result-disclaimer { color:#718096; font-style:italic; margin-top:10px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS & DATA
# =========================================================
@st.cache_resource
def load_resources():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df_path = os.path.join(base, "data", "pre_processed", "pre_processed_data_for_kaggle.csv")
    model_h_path = os.path.join(base, "models", "model_xgb_house.pkl")
    model_a_path = os.path.join(base, "models", "model_xgb_apartment.pkl")
    pipe_h_path = os.path.join(base, "models", "stage3_pipeline_house.pkl")
    pipe_a_path = os.path.join(base, "models", "stage3_pipeline_apartment.pkl")

    df = pd.read_csv(df_path)
    region_counts = df["region"].value_counts().to_dict()

    lookup_cols = [
        "postal_code","locality","province","region","median_income",
        "province_benchmark_m2","region_benchmark_m2","national_benchmark_m2",
        "house_avg_m2_province","apt_avg_m2_province",
        "house_avg_m2_region","apt_avg_m2_region"
    ]

    lookup_df = df[lookup_cols].drop_duplicates("postal_code").set_index("postal_code")

    model_house = joblib.load(model_h_path)
    model_apt = joblib.load(model_a_path)
    pipe_house = joblib.load(pipe_h_path)
    pipe_apt = joblib.load(pipe_a_path)

    return lookup_df, region_counts, model_house, model_apt, pipe_house, pipe_apt


lookup_df, region_counts, model_house, model_apt, pipe_house, pipe_apt = load_resources()

# =========================================================
# HELPERS
# =========================================================
def get_metadata(pc, df):
    if pc in df.index:
        return df.loc[pc].to_dict()
    nearest = df.index[np.abs(df.index - pc).argmin()]
    return df.loc[nearest].to_dict()

def compute_model_confidence(pred, model_type):
    err = 0.167 if model_type.lower() == "house" else 0.09
    return pred*(1-err), pred*(1+err)

# =========================================================
# UNICODE-PDF (FPDF2 + DejaVu)
# =========================================================
def generate_pdf_report(pred, ci_low, ci_high, data, prop_type, subtype, state_label, similar_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Load DejaVu for Unicode (€ included)
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    # Header
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(0, 12, "ImmoEliza Valuation Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(0, 7, "AI Valuation based on Belgian real estate transactions.")
    pdf.ln(4)

    # Main price
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, f"Estimated Value: € {pred:,.0f}", ln=True)

    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, f"Confidence Range: € {ci_low:,.0f} – € {ci_high:,.0f}", ln=True)
    pdf.cell(0, 8, f"Comparable Properties: {similar_count:,}", ln=True)

    # Summary
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Property Summary", ln=True)
    pdf.set_font("DejaVu", "", 12)

    summary = [
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
        f"Energy Consumption: {data['primary_energy_consumption']} kWh/m²",
    ]
    for line in summary:
        pdf.cell(0, 7, line, ln=True)

    # Amenities
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Amenities", ln=True)
    pdf.set_font("DejaVu", "", 12)

    amenities = [
        f"Garage: {'Yes' if data['has_garage'] else 'No'}",
        f"Garden: {'Yes' if data['has_garden'] else 'No'}",
        f"Terrace: {'Yes' if data['has_terrace'] else 'No'}",
        f"Equipped Kitchen: {'Yes' if data['has_equipped_kitchen'] else 'No'}",
        f"Swimming Pool: {'Yes' if data['has_swimming_pool'] else 'No'}",
    ]
    for line in amenities:
        pdf.cell(0, 7, line, ln=True)

    # Location
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Location Insights", ln=True)
    pdf.set_font("DejaVu", "", 12)

    loc = [
        f"Province: {data['province']}",
        f"Region: {data['region']}",
        f"Median Income: € {data['median_income']:,.0f}",
        f"Province Benchmark: € {data['province_benchmark_m2']:,.0f} €/m²",
        f"Region Benchmark: € {data['region_benchmark_m2']:,.0f} €/m²",
        f"National Benchmark: € {data['national_benchmark_m2']:,.0f} €/m²",
    ]
    for line in loc:
        pdf.cell(0, 7, line, ln=True)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 6,
        "Disclaimer: Estimate based on similar properties in your area. "
        "Actual market value may vary."
    )

    return pdf.output(dest="S").encode("latin1")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/AmineSam/immo-eliza-deployment-Amine/main/images/%E2%80%94Pngtree%E2%80%94financial%20investment%20real%20estate%20house_7128805.png",
        width=140,
    )

    st.title("ImmoEliza")

    st.markdown("""
    <div class="sidebar-box">
        This AI-driven valuation tool analyzes thousands of real estate
        transactions in Belgium to estimate property prices.<br><br>
        <b>Models:</b><br>
        🏠 House Model<br>
        🏢 Apartment Model
    </div>
    """, unsafe_allow_html=True)

    st.caption("v1.0.0 | AI Valuation Engine")

# =========================================================
# MAIN FORM
# =========================================================
st.title("Belgian Property Valuation Tool")

st.markdown('<div class="section-header"><span class="section-icon">▸</span> Property Details</div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    prop_type = st.selectbox("Property Type", ["House", "Apartment"], index=None)

    HOUSE = ["residence","villa","mixed building","master house","cottage","bungalow","chalet","mansion"]
    APT = ["apartment","ground floor","penthouse","duplex","studio","loft","triplex","student flat","student housing"]

    if prop_type:
        subtype_list = HOUSE if prop_type == "House" else APT
        subtype_ui = st.selectbox("Subtype", [x.title() for x in subtype_list], index=None)
        prop_subtype = subtype_ui.lower() if subtype_ui else None
    else:
        prop_subtype = None

    postal_code = st.selectbox("Postal Code", sorted(lookup_df.index.tolist()), index=None)

with c2:
    build_year = st.number_input("Build Year", 1800, 2030, None)
    state_ui = st.selectbox(
        "Building State",
        ["New / Recently Renovated", "Good Condition", "Needs Renovation"],
        index=None
    )
    state_map = {"New / Recently Renovated":4, "Good Condition":2, "Needs Renovation":0}
    state_val = state_map.get(state_ui, 2)

# SIZE & ENERGY
st.markdown('<div class="section-header"><span class="section-icon">▸</span> Size & Energy</div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1: area = st.slider("Living Area (m²)", 15, 500, 120)
with c2: energy = st.slider("Primary Energy Consumption", 0, 500, 200)

# Rooms
st.markdown('<div class="section-header"><span class="section-icon">▸</span> Rooms & Layout</div>',
            unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
rooms = r1.number_input("Bedrooms", 0, 10, 3)
bathrooms = r2.number_input("Bathrooms", 0, 5, 1)
toilets = r3.number_input("Toilets", 0, 5, 2)
facades = r4.number_input("Facades", 0, 4, 2)

# Amenities
st.markdown('<div class="section-header"><span class="section-icon">▸</span> Amenities</div>',
            unsafe_allow_html=True)

a1, a2 = st.columns(2)
with a1:
    has_garage = st.toggle("Garage")
    has_garden = st.toggle("Garden")
    has_terrace = st.toggle("Terrace")

with a2:
    has_kitchen = st.toggle("Equipped Kitchen")
    has_pool = st.toggle("Swimming Pool")

st.markdown("---")

# BUTTON
center = st.columns([1,2,1])
submitted = center[1].button("Estimate Price")

# =========================================================
# PREDICTION HANDLING
# =========================================================
if submitted:

    # validate
    if not prop_type or not prop_subtype or not postal_code or not build_year or not state_ui:
        st.error("Please fill in all required fields.")
        st.stop()

    with st.spinner("Analyzing market data..."):

        metadata = get_metadata(postal_code, lookup_df)
        region_name = metadata["region"]
        similar_count = region_counts.get(region_name, 500)

        data = {
            "property_type": prop_type,
            "property_subtype": prop_subtype,
            "postal_code": postal_code,
            "locality": metadata["locality"],
            "area": area,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "toilets": toilets,
            "primary_energy_consumption": energy,
            "state": state_val,
            "build_year": build_year,
            "facades_number": facades,
            "has_garage": int(has_garage),
            "has_garden": int(has_garden),
            "has_terrace": int(has_terrace),
            "has_equipped_kitchen": int(has_kitchen),
            "has_swimming_pool": int(has_pool),
            "median_income": metadata["median_income"],
            "province": metadata["province"],
            "region": metadata["region"],
            "province_benchmark_m2": metadata["province_benchmark_m2"],
            "region_benchmark_m2": metadata["region_benchmark_m2"],
            "national_benchmark_m2": metadata["national_benchmark_m2"],
            "house_avg_m2_province": metadata["house_avg_m2_province"],
            "apt_avg_m2_province": metadata["apt_avg_m2_province"],
            "house_avg_m2_region": metadata["house_avg_m2_region"],
            "apt_avg_m2_region": metadata["apt_avg_m2_region"],
        }

        # Select model + pipeline
        pipeline = pipe_house if prop_type == "House" else pipe_apt
        model = model_house if prop_type == "House" else model_apt

        # Stage3 transform
        df_in = pd.DataFrame([data])
        df_s3 = transform_stage3(df_in, pipeline)

        RED_FEATS = [
            "area","postal_code_te_price","locality_te_price","bathrooms","rooms",
            "primary_energy_consumption","state","province_benchmark_m2","postal_code",
            "region_benchmark_m2","property_subtype_te_price","apt_avg_m2_region",
            "toilets","property_type_te_price","median_income","build_year",
            "house_avg_m2_province","has_garage","apt_avg_m2_province","has_garden",
            "has_terrace","facades_number","has_swimming_pool",
            "house_avg_m2_region","has_equipped_kitchen"
        ]

        X = df_s3[[c for c in RED_FEATS if c in df_s3.columns]]
        pred = float(model.predict(X)[0])
        ci_low, ci_high = compute_model_confidence(pred, prop_type)

        # =========================================================
        # RESULT CARD
        # =========================================================
        st.markdown(f"""
<div class="result-card">
    <div class="result-title">Estimated Property Value</div>
    <div class="result-price">€ {pred:,.0f}</div>

    <div style="font-size:1.1rem; margin-top:10px; color:#2c5282;">
        <b>Confidence Range:</b><br>
        € {ci_low:,.0f} – € {ci_high:,.0f}
    </div>

    <div class="result-subline">
        Based on <strong>{similar_count:,}</strong> similar properties in {region_name}.
    </div>

    <div class="result-disclaimer">
        Disclaimer: Estimate based on similar properties in your area.
    </div>
</div>
""", unsafe_allow_html=True)

        # =========================================================
        # PDF
        # =========================================================
        pdf_bytes = generate_pdf_report(
            pred, ci_low, ci_high, data,
            prop_type, prop_subtype, state_ui, similar_count
        )

        st.download_button(
            "Download Valuation PDF",
            data=pdf_bytes,
            file_name="immoeliza_valuation.pdf",
            mime="application/pdf"
        )

        # =========================================================
        # BENCHMARK CHART
        # =========================================================
        st.markdown("---")
        st.markdown("### Price Benchmarks")

        prov_price = metadata["province_benchmark_m2"] * area
        reg_price = metadata["region_benchmark_m2"] * area

        chart_df = pd.DataFrame({
            "Category": ["Predicted", "Province Avg", "Region Avg"],
            "Price": [pred, prov_price, reg_price]
        })

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Category", sort=None),
            y="Price",
            tooltip=[alt.Tooltip("Price", format=",.0f")]
        )
        st.altair_chart(chart, use_container_width=True)

        # =========================================================
        # LOCATION INSIGHTS
        # =========================================================
        with st.expander("See Location Insights", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Province", metadata["province"])
            c2.metric("Region", metadata["region"])
            c3.metric("Locality", metadata["locality"])

            st.markdown("#### Market Benchmarks (€/m²)")

            b1, b2, b3 = st.columns(3)
            b1.metric("Province Avg", f"€ {metadata['province_benchmark_m2']:,.0f}")
            b2.metric("Region Avg", f"€ {metadata['region_benchmark_m2']:,.0f}")
            b3.metric("National Avg", f"€ {metadata['national_benchmark_m2']:,.0f}")

"""
Streamlit House Price Prediction App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
    
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    }
    .hero-card h1 { color: white; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero-card p  { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin-top: 0.5rem; }

    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-label { color: rgba(255,255,255,0.6); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: #00f2fe; font-size: 2.2rem; font-weight: 700; }

    .predict-btn .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white; font-weight: 700; font-size: 1.1rem;
        border: none; border-radius: 12px;
        padding: 0.8rem 2rem; width: 100%;
        box-shadow: 0 8px 25px rgba(245,87,108,0.4);
        transition: all 0.3s;
    }
    .predict-btn .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 35px rgba(245,87,108,0.5); }

    [data-testid="stSidebar"] { background: rgba(15,12,41,0.95); border-right: 1px solid rgba(255,255,255,0.1); }
    [data-testid="stSidebar"] * { color: white !important; }

    .price-result {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 45px rgba(56,239,125,0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%,100% { box-shadow: 0 15px 45px rgba(56,239,125,0.3); }
        50%      { box-shadow: 0 20px 60px rgba(56,239,125,0.5); }
    }
    .price-result h2 { color: white; font-size: 3rem; font-weight: 700; margin: 0; }
    .price-result p  { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.4rem 0 0; }

    .stSlider > div > div { background: rgba(255,255,255,0.1) !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: rgba(255,255,255,0.8) !important; }

    hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Gradient Boosting": "models/gradient_boosting_final.pkl",
        "Random Forest":     "models/random_forest_final.pkl",
        "Linear Regression": "models/linear_regression.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    scaler = None
    for scaler_path in ['models/scaler_phase2.pkl', 'models/scaler.pkl']:
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            break

    features = None
    for feat_path in ['models/feature_names_phase2.json', 'models/feature_names.json']:
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                features = json.load(f)
            break

    return models, scaler, features


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Input Features")
    st.markdown("Adjust the sliders to describe your house:")
    st.markdown("---")

    # Key features — will be mapped into the feature vector
    overall_qual  = st.slider("Overall Quality (1–10)", 1, 10, 6, help="Overall material and finish quality")
    gr_liv_area   = st.number_input("Above-Ground Living Area (sq ft)", 500, 6000, 1500)
    garage_cars   = st.slider("Garage Capacity (cars)", 0, 4, 2)
    year_built    = st.slider("Year Built", 1870, 2024, 2000)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    full_bath     = st.slider("Full Bathrooms", 0, 4, 2)
    bedrooms      = st.slider("Bedrooms Above Grade", 0, 8, 3)
    fireplaces    = st.slider("Fireplaces", 0, 3, 1)
    lot_area      = st.number_input("Lot Area (sq ft)", 1000, 200000, 8000)
    has_pool      = st.checkbox("Has Pool 🏊", value=False)

    st.markdown("---")
    model_choice = st.selectbox(
        "Select Prediction Model",
        ["Gradient Boosting", "Random Forest", "Linear Regression"],
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.5); font-size:0.75rem; text-align:center'>
    ML PBL Project<br>House Price Predictor v2.0
    </div>""", unsafe_allow_html=True)


# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-card'>
    <h1>🏠 House Price Predictor</h1>
    <p>Powered by Gradient Boosting & Random Forest — ML PBL Phase 2 Showcase</p>
</div>
""", unsafe_allow_html=True)

# ─── Model Status ─────────────────────────────────────────────────────────────
models, scaler, features = load_models()

col1, col2, col3 = st.columns(3)
with col1:
    status = "✅ Ready" if models else "⚠️ Demo Mode"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Model Status</div>
        <div class='metric-value' style='font-size:1.3rem'>{status}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Active Model</div>
        <div class='metric-value' style='font-size:1.1rem; color:#f5576c'>{model_choice}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Models Available</div>
        <div class='metric-value'>{len(models) if models else 'Demo'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Prediction Logic ─────────────────────────────────────────────────────────
def make_prediction(input_vals: dict) -> float:
    """Predict price. Falls back to a regression formula if models aren't trained yet."""
    
    if models and scaler and features:
        # Build feature vector aligned to training features
        X_input = pd.DataFrame([{f: 0 for f in features}])

        # Map known inputs to feature names (best-effort)
        mapping = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars,
            'YearBuilt': year_built,
            'TotalBsmtSF': total_bsmt_sf,
            'FullBath': full_bath,
            'BedroomAbvGr': bedrooms,
            'Fireplaces': fireplaces,
            'LotArea': lot_area,
            # Engineered features
            'HouseAge': 2024 - year_built,
            'TotalSF': total_bsmt_sf + gr_liv_area,
            'TotalBath': full_bath + 0.5,
            'HasPool': int(has_pool),
            'HasGarage': int(garage_cars > 0),
            'HasFireplace': int(fireplaces > 0),
        }

        for col, val in mapping.items():
            if col in X_input.columns:
                X_input[col] = val

        X_scaled = scaler.transform(X_input)
        model = models[model_choice]
        pred = model.predict(X_scaled)[0]
        # If log-transformed output
        if pred < 20:   # log-space
            pred = np.expm1(pred)
        return pred

    else:
        # Demo formula (approximate linear model)
        price = (
            overall_qual     * 12_000 +
            gr_liv_area      * 60 +
            garage_cars      * 8_000 +
            (2024 - year_built) * -300 +
            total_bsmt_sf    * 30 +
            full_bath        * 6_000 +
            fireplaces       * 5_000 +
            lot_area         * 0.5 +
            (15_000 if has_pool else 0) +
            70_000            # intercept
        )
        return max(price, 50_000)


# ─── Predict Button ───────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🔢 Your House Summary")
    summary_data = {
        "Feature": ["Overall Quality", "Living Area", "Garage Cars", "Year Built",
                    "Basement (sq ft)", "Full Baths", "Bedrooms", "Fireplaces", "Lot Area", "Has Pool"],
        "Value":   [f"{overall_qual}/10", f"{gr_liv_area:,} sq ft", f"{garage_cars} cars",
                    str(year_built), f"{total_bsmt_sf:,} sq ft", str(full_bath), str(bedrooms),
                    str(fireplaces), f"{lot_area:,} sq ft", "Yes" if has_pool else "No"],
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
    
    st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict House Price", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("### 📊 Price Estimate")

    if predict_clicked:
        with st.spinner("Running model inference..."):
            price = make_prediction({})
            price_low = price * 0.92
            price_high = price * 1.08

        st.markdown(f"""
        <div class='price-result'>
            <p>Estimated House Price</p>
            <h2>${price:,.0f}</h2>
            <p>Range: ${price_low:,.0f} — ${price_high:,.0f}</p>
        </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')

        # Price bands
        bands = [
            (0, 150_000, '#e74c3c', 'Budget'),
            (150_000, 300_000, '#e67e22', 'Affordable'),
            (300_000, 500_000, '#f1c40f', 'Mid-Range'),
            (500_000, 750_000, '#2ecc71', 'Premium'),
            (750_000, 1_200_000, '#3498db', 'Luxury'),
        ]
        for lo, hi, color, label in bands:
            ax.barh(0, hi - lo, left=lo, height=0.4, color=color, alpha=0.7, label=label)

        clamped = min(max(price, 0), 1_200_000)
        ax.axvline(clamped, color='white', linewidth=3, label=f'${price:,.0f}')
        ax.scatter([clamped], [0], color='white', s=200, zorder=5)

        ax.set_yticks([])
        ax.set_xlim(0, 1_200_000)
        ax.tick_params(colors='white')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax.set_title('Price Band Indicator', color='white', fontsize=11, pad=12)
        ax.legend(loc='upper right', fontsize=8, facecolor='#24243e', labelcolor='white', framealpha=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor((1, 1, 1, 0.1))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if not models:
            st.info("💡 Tip: Run the notebooks first to train and save models, then relaunch the app for ML-powered predictions.")

    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:rgba(255,255,255,0.4)'>
            <div style='font-size:4rem'>🏡</div>
            <p>Adjust the sliders on the left, then click <strong>Predict</strong></p>
        </div>""", unsafe_allow_html=True)


# ─── Feature Impact Visualization ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Feature Impact on Your Price Estimate")

impact_features = {
    "Overall Quality": overall_qual * 12_000,
    "Living Area":     gr_liv_area * 60,
    "Basement Area":   total_bsmt_sf * 30,
    "Age Penalty":     -(2024 - year_built) * 300,
    "Garage":          garage_cars * 8_000,
    "Bathrooms":       full_bath * 6_000,
    "Pool Bonus":      15_000 if has_pool else 0,
    "Fireplaces":      fireplaces * 5_000,
}

fig2, ax2 = plt.subplots(figsize=(12, 5))
fig2.set_facecolor('#0f0c29')
ax2.set_facecolor('#0f0c29')

items = sorted(impact_features.items(), key=lambda x: x[1])
names = [k for k, _ in items]
vals  = [v for _, v in items]
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in vals]

bars = ax2.barh(names, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.4)
ax2.set_xlabel('Price Contribution ($)', color='white', fontsize=10)
ax2.set_title('Estimated Feature Contribution to Price', color='white', fontsize=12, fontweight='bold', pad=10)
ax2.tick_params(colors='white', labelsize=9)
ax2.axvline(0, color='white', linewidth=0.8, alpha=0.5)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

for sp in ax2.spines.values():
    sp.set_edgecolor((1, 1, 1, 0.2))

ax2.bar_label(bars, labels=[f'${v/1000:+.0f}k' for v in vals], padding=4, color='white', fontsize=8)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.4); font-size:0.8rem; padding: 1rem 0'>
    🏠 House Price Predictor · ML PBL Project · Built with Streamlit + scikit-learn<br>
    Phase 1: Linear Regression Baseline · Phase 2: Gradient Boosting & Random Forest Ensemble
</div>""", unsafe_allow_html=True)

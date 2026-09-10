import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    load_models_and_artifacts, 
    l100km_to_mpg, 
    get_efficiency_category, 
    calculate_annual_fuel_cost, 
    get_custom_css
)

# Page Setup
st.set_page_config(
    page_title="AI Vehicle Fuel Consumption Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_all_artifacts():
    return load_models_and_artifacts()

@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "vehicle_fuel_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

artifacts = load_all_artifacts()
df_data = load_dataset()

regressors = artifacts["regressors"]
preprocessor = artifacts["preprocessor"]
metrics = artifacts["metrics"]

# Header
st.markdown("""
<div class="header-banner">
    <div class="header-title">🚗 AI-Based Vehicle Fuel Consumption Prediction System</div>
    <div class="header-subtitle">Predicting Fuel Economy (L/100 km & MPG) using Automotive Specs & Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("## ⚙️ Vehicle Specifications")

# Preset Selector
preset = st.sidebar.radio(
    "⚡ Vehicle Presets:",
    ["Custom Specs", "🏎️ Compact Eco Hatch", "🚗 Family Sedan", "🚙 Heavy SUV", "⚡ Hybrid Commuter"],
    index=0
)

if preset == "🏎️ Compact Eco Hatch":
    d_es, d_cyl, d_hp, d_weight, d_acc, d_yr, d_ft, d_tr = 1.2, 3, 85, 1050, 11.5, 2022, "Petrol", "Manual"
elif preset == "🚗 Family Sedan":
    d_es, d_cyl, d_hp, d_weight, d_acc, d_yr, d_ft, d_tr = 2.0, 4, 150, 1420, 8.8, 2021, "Petrol", "Automatic"
elif preset == "🚙 Heavy SUV":
    d_es, d_cyl, d_hp, d_weight, d_acc, d_yr, d_ft, d_tr = 3.5, 6, 295, 2180, 6.8, 2023, "Petrol", "Automatic"
elif preset == "⚡ Hybrid Commuter":
    d_es, d_cyl, d_hp, d_weight, d_acc, d_yr, d_ft, d_tr = 1.8, 4, 122, 1360, 10.2, 2023, "Hybrid", "Automatic"
else: # Custom
    d_es, d_cyl, d_hp, d_weight, d_acc, d_yr, d_ft, d_tr = 2.4, 4, 175, 1550, 8.2, 2020, "Petrol", "Automatic"

# Inputs
st.sidebar.markdown("### 🎛️ Engine & Chassis Parameters")
engine_size = st.sidebar.slider("Engine Displacement (L)", 1.0, 6.0, float(d_es), 0.1)
cylinders = st.sidebar.selectbox("Number of Cylinders", [3, 4, 6, 8, 12], index=[3, 4, 6, 8, 12].index(d_cyl))
horsepower = st.sidebar.slider("Horsepower (HP)", 70, 500, int(d_hp), 5)
weight = st.sidebar.slider("Vehicle Weight (kg)", 900, 2800, int(d_weight), 20)
acceleration = st.sidebar.slider("0-100 km/h Acceleration (sec)", 3.5, 15.0, float(d_acc), 0.1)
model_year = st.sidebar.slider("Model Year", 2012, 2024, int(d_yr), 1)

st.sidebar.markdown("### ⛽ Fuel & Powertrain")
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Plug-in Hybrid"], index=["Petrol", "Diesel", "Hybrid", "Plug-in Hybrid"].index(d_ft))
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual"], index=["Automatic", "Manual"].index(d_tr))

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("🤖 ML Model Engine", list(regressors.keys()), index=2)

# Input DataFrame Construction
input_dict = {
    'Engine Size': engine_size,
    'Cylinders': cylinders,
    'Horsepower': horsepower,
    'Vehicle Weight': weight,
    'Acceleration': acceleration,
    'Model Year': model_year,
    'Fuel Type': fuel_type,
    'Transmission': transmission
}

input_df = pd.DataFrame([input_dict])
input_proc = preprocessor.transform(input_df)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🚘 Live Fuel Estimator", 
    "📊 Automotive EDA & Insights", 
    "🤖 Model Benchmarks & Importance", 
    "⛽ Eco-Driving Guide"
])

# TAB 1: LIVE FUEL ESTIMATOR
with tab1:
    col_l, col_r = st.columns([1, 1])
    
    # Model Prediction
    model = regressors[selected_model_name]
    pred_l100km = round(float(model.predict(input_proc)[0]), 2)
    pred_mpg = l100km_to_mpg(pred_l100km)
    efficiency_info = get_efficiency_category(pred_l100km)
    
    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⛽ Fuel Economy Prediction")
        st.caption(f"Engine Model: **{selected_model_name}**")
        
        # Plotly Gauge Chart for Fuel Consumption (L/100 km)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_l100km,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Estimated Fuel Consumption (L/100 km)", 'font': {'size': 18, 'color': '#f9fafb'}},
            gauge = {
                'axis': {'range': [2, 18], 'tickwidth': 1, 'tickcolor': "#9ca3af"},
                'bar': {'color': "#2563eb"},
                'bgcolor': "rgba(17, 24, 39, 0.5)",
                'borderwidth': 2,
                'bordercolor': "#374151",
                'steps': [
                    {'range': [2, 6.0], 'color': 'rgba(16, 185, 129, 0.35)'},
                    {'range': [6.0, 9.0], 'color': 'rgba(245, 158, 11, 0.35)'},
                    {'range': [9.0, 18.0], 'color': 'rgba(239, 68, 68, 0.35)'}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Metric Callouts
        m1, m2 = st.columns(2)
        m1.metric("Fuel Economy (L/100 km)", f"{pred_l100km} L/100km")
        m2.metric("Equivalent Miles Per Gallon", f"{pred_mpg} MPG")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Efficiency Classification")
        
        st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span style="font-size: 1.1rem; color: #9ca3af; display: block; margin-bottom: 0.6rem;">Rating Category:</span>
            <span class="status-badge {efficiency_info['badge_class']}">{efficiency_info['icon']} {efficiency_info['category']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Automotive Assessment**: {efficiency_info['summary']}")
        
        st.markdown("#### Vehicle Power-to-Weight Spec:")
        hp_per_tonne = round((horsepower / (weight / 1000.0)), 1)
        st.markdown(f"- **Power-to-Weight Ratio**: `{hp_per_tonne} HP / Tonne`")
        st.markdown(f"- **Specific Output**: `{round(horsepower / engine_size, 1)} HP / Liter`")
        st.markdown('</div>', unsafe_allow_html=True)

    # LOWER SECTION: Annual Financial & Environmental Calculator
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Financial & Environmental Cost Calculator")
    
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c1:
        annual_dist = st.number_input("Annual Driving Distance (km)", 2000, 100000, 15000, 1000)
    with col_c2:
        fuel_price = st.number_input("Fuel Price per Liter ($ / L)", 0.50, 5.00, 1.60, 0.05)
    with col_c3:
        cost_info = calculate_annual_fuel_cost(pred_l100km, annual_dist, fuel_price)
        st.markdown(f"**Annual Fuel Needed**: `{cost_info['total_liters']} Liters`")
        
    f1, f2, f3 = st.columns(3)
    f1.metric("Est. Annual Fuel Cost", f"${cost_info['annual_cost']:,.2f}")
    f2.metric("Est. Annual CO₂ Footprint", f"{cost_info['annual_co2_tons']} Tons CO₂")
    f3.metric("Est. Monthly Fuel Bill", f"${round(cost_info['annual_cost'] / 12, 2)} / month")
    st.markdown('</div>', unsafe_allow_html=True)


# TAB 2: AUTOMOTIVE EDA & INSIGHTS
with tab2:
    if df_data is not None:
        st.markdown("### 📊 Dataset Exploration & Physical Relationships")
        
        c_e1, c_e2 = st.columns([1, 2])
        with c_e1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Vehicle Dataset Sample")
            st.dataframe(df_data.head(8), height=260)
            st.markdown(f"**Total Samples**: `{len(df_data)}` | **Features**: `{len(df_data.columns)}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_e2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Feature Correlation Heatmap")
            num_cols = ['Engine Size', 'Cylinders', 'Horsepower', 'Vehicle Weight', 'Acceleration', 'Model Year', 'Fuel Consumption']
            corr = df_data[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale="Cividis", title="Feature Correlations")
            fig_corr.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10), font=dict(color="#f9fafb"))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### 📈 Interactive Parameter Relationship Scatter")
        c_s1, c_s2 = st.columns([1, 2])
        with c_s1:
            x_var = st.selectbox("X-Axis Feature", ['Vehicle Weight', 'Engine Size', 'Horsepower', 'Acceleration'], index=0)
            y_var = st.selectbox("Y-Axis Feature", ['Fuel Consumption', 'Horsepower', 'Acceleration'], index=0)
            color_var = st.selectbox("Color By Categorical", ['Fuel Type', 'Transmission', 'Cylinders'], index=0)
            
        with c_s2:
            fig_scatter = px.scatter(
                df_data, x=x_var, y=y_var, color=color_var,
                size='Engine Size' if y_var != 'Engine Size' else 'Horsepower',
                hover_data=['Model Year', 'Transmission'],
                title=f"{y_var} vs {x_var} grouped by {color_var}"
            )
            fig_scatter.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"))
            st.plotly_chart(fig_scatter, use_container_width=True)


# TAB 3: MODEL BENCHMARKS & FEATURE IMPORTANCE
with tab3:
    st.markdown("### 🤖 Model Performance Benchmark Matrix")
    
    if metrics:
        reg_df = pd.DataFrame(metrics["Regression Metrics"]).T
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Regression Model Evaluation (Test Set)")
            st.dataframe(reg_df.style.highlight_min(axis=0, subset=['MAE', 'RMSE'], color='#065f46')
                                    .highlight_max(axis=0, subset=['R2'], color='#065f46'), use_container_width=True)
            
            fig_bar = px.bar(
                reg_df.reset_index(), x='index', y=['MAE', 'RMSE'], barmode='group',
                title="MAE & RMSE Error (Lower is Better)",
                labels={'index': 'Model', 'value': 'L/100 km'}
            )
            fig_bar.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_m2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### R² Variance Explained Score")
            fig_r2 = px.bar(
                reg_df.reset_index(), x='index', y='R2',
                color='index', title="R² Accuracy Score (Higher is Better)",
                labels={'index': 'Model', 'R2': 'R² Score'}
            )
            fig_r2.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"), showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### 🌲 Feature Importance Breakdown (Random Forest)")
        fi_dict = metrics.get("Feature Importance", {})
        if fi_dict:
            fi_df = pd.DataFrame(list(fi_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Feature Contribution to Fuel Consumption")
            fig_fi.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"))
            st.plotly_chart(fig_fi, use_container_width=True)


# TAB 4: ECO-DRIVING GUIDE
with tab4:
    st.markdown("### ⛽ Vehicle Fuel Efficiency & Eco-Driving Advisory")
    
    st.markdown(r"""
    #### 🚗 Key Mechanical Factors Influencing Fuel Economy

    1. **Vehicle Weight & Inertia ($\mathbf{E \propto m \cdot a}$)**
       - Accelerating a heavier vehicle requires significantly more mechanical kinetic energy.
       - *Rule of Thumb*: Every **100 kg weight reduction** improves fuel economy by **2% to 4%**.

    2. **Engine Displacement & Pumping Losses ($\mathbf{V_d}$)**
       - Larger engine displacements consume more fuel even at idle due to internal friction and throttling intake losses.
       - Turbocharging smaller engines (downsizing) allows high power outputs while maintaining low fuel consumption under light loads.

    3. **Hybrid & Regenerative Powertrains**
       - Hybrids capture braking kinetic energy to recharge traction batteries, drastically reducing city fuel consumption in stop-and-go traffic.

    ---

    #### 💡 Practical Eco-Driving Tips for Automotive Engineers & Drivers

    - **Proper Tire Inflation**: Low tire pressure increases rolling resistance coefficient ($C_{rr}$), increasing fuel usage by up to 3%.
    - **Maintain Smooth Acceleration**: Rapid acceleration operates the engine in rich fuel mixture zones, consuming extra fuel.
    - **Aerodynamic Drag Management**: At highway speeds ($> 80 \text{ km/h}$), aerodynamic drag ($F_d = \frac{1}{2} \rho C_d A v^2$) dominates fuel consumption. Keep windows closed and remove unused roof racks.
    """)

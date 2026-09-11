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
from utils.gps_engine import (
    PRESET_ROUTES,
    generate_route_telemetry,
    predict_gps_route_fuel,
    generate_eco_route_alternatives,
    parse_custom_gps_csv,
    haversine_distance
)

# Page Setup
st.set_page_config(
    page_title="AI Vehicle Fuel Consumption & GPS Estimator",
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
    <div class="header-title">🚗 AI Vehicle Fuel Consumption & GPS Navigation System</div>
    <div class="header-subtitle">Machine Learning Fuel Prediction, GPS Route Telemetry, Elevation Physics & Eco-Routing Optimization</div>
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

# Base Model Prediction
model = regressors[selected_model_name]
pred_l100km = round(float(model.predict(input_proc)[0]), 2)
pred_mpg = l100km_to_mpg(pred_l100km)
efficiency_info = get_efficiency_category(pred_l100km)

# Tabs Navigation
tab1, tab_gps, tab2, tab3, tab4 = st.tabs([
    "🚘 Live Fuel Estimator", 
    "🛰️ GPS Route & Trip Predictor",
    "📊 Automotive EDA & Insights", 
    "🤖 Model Benchmarks & Importance", 
    "⛽ Eco-Driving Guide"
])

# ==============================================================================
# TAB 1: LIVE FUEL ESTIMATOR
# ==============================================================================
with tab1:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⛽ Fuel Economy Prediction")
        st.caption(f"Engine Model: **{selected_model_name}**")
        
        # Plotly Gauge Chart for Fuel Consumption (L/100 km)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_l100km,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Standard Cycle Fuel Consumption (L/100 km)", 'font': {'size': 17, 'color': '#f9fafb'}},
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
        fuel_price = st.number_input("Fuel Price per Liter ($ / L)", 0.50, 5.00, 1.60, 0.05, key="annual_fuel_price")
    with col_c3:
        cost_info = calculate_annual_fuel_cost(pred_l100km, annual_dist, fuel_price)
        st.markdown(f"**Annual Fuel Needed**: `{cost_info['total_liters']} Liters`")
        
    f1, f2, f3 = st.columns(3)
    f1.metric("Est. Annual Fuel Cost", f"${cost_info['annual_cost']:,.2f}")
    f2.metric("Est. Annual CO₂ Footprint", f"{cost_info['annual_co2_tons']} Tons CO₂")
    f3.metric("Est. Monthly Fuel Bill", f"${round(cost_info['annual_cost'] / 12, 2)} / month")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: GPS ROUTE & TRIP PREDICTOR (NEW!)
# ==============================================================================
with tab_gps:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <h3 style="margin: 0; color: #60a5fa;"><span class="pulse-dot"></span>🛰️ Real-Time GPS Route Telemetry & Fuel Consumption Predictor</h3>
            <p style="margin: 0.2rem 0 0 0; color: #9ca3af; font-size: 0.95rem;">
                Combines vehicle specs with physical GPS terrain elevation, road speeds, aerodynamic drag, and traffic patterns.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Route Configuration Box
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🗺️ Route Setup & Driving Profile")
    
    route_mode = st.radio(
        "Select GPS Input Mode:",
        ["📍 Predefined Real-World Routes", "🎯 Custom GPS Coordinates & Route Specs", "📂 Upload Custom GPS Telemetry Log (CSV)"],
        horizontal=True
    )
    
    col_r1, col_r2, col_r3 = st.columns([1.5, 1, 1])
    
    custom_params = {}
    df_telemetry = None
    uploaded_mode = False
    
    if route_mode == "📍 Predefined Real-World Routes":
        with col_r1:
            selected_route_key = st.selectbox(
                "Choose Highway / Scenic GPS Route:",
                list(PRESET_ROUTES.keys()),
                index=0
            )
            route_info = PRESET_ROUTES[selected_route_key]
            st.caption(f"🛣️ **Route Summary**: {route_info['description']}")
            
        with col_r2:
            driving_style = st.selectbox(
                "Driving Behavior / Aggressiveness:",
                ["🌿 Eco-Conscious", "🚗 Normal Balanced", "🏎️ Dynamic / Aggressive"],
                index=1
            )
            traffic_condition = st.select_slider(
                "Traffic Flow Condition:",
                options=["🟢 Free Flow", "🟡 Moderate Traffic", "🔴 Heavy Congestion"],
                value="🟡 Moderate Traffic"
            )
            
        with col_r3:
            gps_fuel_price = st.number_input(
                "Fuel Price ($ / Liter):",
                min_value=0.50, max_value=5.00, value=1.60, step=0.05, key="gps_fuel_price"
            )
            nominal_km = route_info['nominal_distance_km']
            st.markdown(f"- **Nominal Distance**: `{nominal_km} km` ({round(nominal_km*0.621371, 1)} mi)")
            st.markdown(f"- **Terrain Category**: `{route_info['terrain_type']}`")
            st.markdown(f"- **Elevation Change**: `{route_info['base_elevation_start']}m ➔ {route_info['base_elevation_end']}m`")
            
        # Generate telemetry
        df_telemetry = generate_route_telemetry(selected_route_key)
        
    elif route_mode == "🎯 Custom GPS Coordinates & Route Specs":
        with col_r1:
            c_orig_name = st.text_input("Origin Location / City", "San Francisco, CA")
            c_dest_name = st.text_input("Destination Location / City", "Sacramento, CA")
            c_dist = st.number_input("Total Trip Distance (km)", min_value=10.0, max_value=2500.0, value=140.0, step=10.0)
            
        with col_r2:
            c_start_lat = st.number_input("Start Latitude", value=37.7749, format="%.4f")
            c_start_lon = st.number_input("Start Longitude", value=-122.4194, format="%.4f")
            c_end_lat = st.number_input("End Latitude", value=38.5816, format="%.4f")
            c_end_lon = st.number_input("End Longitude", value=-121.4944, format="%.4f")
            
        with col_r3:
            c_terrain = st.selectbox("Terrain Profile", ["Rolling Hills", "Mountainous", "Coastal Flat & Lowlands"], index=0)
            c_elev_gain = st.number_input("Peak Elevation (meters)", min_value=0.0, max_value=5000.0, value=120.0, step=50.0)
            c_speed = st.slider("Target Cruising Speed (km/h)", 40, 130, 95, 5)
            driving_style = st.selectbox("Driving Behavior:", ["🌿 Eco-Conscious", "🚗 Normal Balanced", "🏎️ Dynamic / Aggressive"], index=1, key="cust_style")
            gps_fuel_price = st.number_input("Fuel Price ($ / L):", 0.50, 5.00, 1.60, 0.05, key="cust_fuel_price")

        custom_params = {
            "start_lat": c_start_lat,
            "start_lon": c_start_lon,
            "end_lat": c_end_lat,
            "end_lon": c_end_lon,
            "distance_km": c_dist,
            "elev_start": 10.0,
            "elev_end": 50.0,
            "elev_peak": c_elev_gain,
            "avg_speed": c_speed,
            "terrain": c_terrain
        }
        df_telemetry = generate_route_telemetry("custom", custom_params=custom_params)
        selected_route_key = f"Custom: {c_orig_name} ➔ {c_dest_name}"
        
    else: # Upload CSV
        with col_r1:
            uploaded_file = st.file_uploader(
                "Upload GPS Route CSV (Must include 'lat' & 'lon', optional 'elevation', 'speed'):",
                type=['csv']
            )
            # Sample CSV download button
            sample_df = pd.DataFrame({
                'latitude': [37.7749, 37.8044, 37.8715, 38.0000, 38.2500, 38.5816],
                'longitude': [-122.4194, -122.2711, -122.2730, -122.1000, -121.8000, -121.4944],
                'elevation': [15, 45, 120, 80, 40, 25],
                'speed_kmh': [40, 95, 105, 110, 100, 50]
            })
            st.download_button(
                "📥 Download Sample GPS CSV Template",
                sample_df.to_csv(index=False),
                "sample_gps_route.csv",
                "text/csv"
            )
            
        with col_r2:
            driving_style = st.selectbox("Driving Behavior:", ["🌿 Eco-Conscious", "🚗 Normal Balanced", "🏎️ Dynamic / Aggressive"], index=1, key="upl_style")
        with col_r3:
            gps_fuel_price = st.number_input("Fuel Price ($ / L):", 0.50, 5.00, 1.60, 0.05, key="upl_fuel_price")

        if uploaded_file is not None:
            try:
                df_telemetry = parse_custom_gps_csv(uploaded_file)
                selected_route_key = f"Uploaded File: {uploaded_file.name}"
                uploaded_mode = True
                st.success(f"✅ Successfully ingested `{len(df_telemetry)}` GPS track waypoints!")
            except Exception as e:
                st.error(f"❌ Error parsing CSV: {e}")
                df_telemetry = None
        else:
            st.info("👆 Please upload a GPS CSV file or download the sample template above to view results.")

    st.markdown('</div>', unsafe_allow_html=True)

    # If telemetry is ready, run predictions and render dashboards
    if df_telemetry is not None and len(df_telemetry) > 0:
        # Run Physical GPS Prediction Engine
        df_results, summary = predict_gps_route_fuel(
            df_telemetry=df_telemetry,
            vehicle_specs=input_dict,
            base_model_l100km=pred_l100km,
            driving_style=driving_style,
            fuel_price_per_l=gps_fuel_price
        )

        # ----------------------------------------------------------------------
        # Top GPS KPI Summary Metrics
        # ----------------------------------------------------------------------
        st.markdown("#### 📊 Predicted Trip Telemetry & Fuel Summary")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        with k1:
            st.markdown("""
            <div class="telemetry-card">
                <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">🛣️ Total Distance</div>
                <div class="telemetry-val">{} <span style="font-size: 1rem; color: #9ca3af;">km</span></div>
                <div style="color: #6ee7b7; font-size: 0.8rem; margin-top: 4px;">{} miles</div>
            </div>
            """.format(summary["total_distance_km"], summary["total_distance_miles"]), unsafe_allow_html=True)

        with k2:
            st.markdown("""
            <div class="telemetry-card">
                <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">⛽ Fuel Required</div>
                <div class="telemetry-val" style="color: #fbbf24;">{} <span style="font-size: 1rem; color: #9ca3af;">L</span></div>
                <div style="color: #fde68a; font-size: 0.8rem; margin-top: 4px;">{} US Gallons</div>
            </div>
            """.format(summary["total_fuel_liters"], summary["total_fuel_gallons"]), unsafe_allow_html=True)

        with k3:
            st.markdown("""
            <div class="telemetry-card">
                <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">💵 Est. Trip Cost</div>
                <div class="telemetry-val" style="color: #34d399;">${:,.2f}</div>
                <div style="color: #9ca3af; font-size: 0.8rem; margin-top: 4px;">@ ${:.2f} / L</div>
            </div>
            """.format(summary["total_cost_usd"], summary["fuel_price_per_l"]), unsafe_allow_html=True)

        with k4:
            st.markdown("""
            <div class="telemetry-card">
                <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">🌿 Total CO₂ Footprint</div>
                <div class="telemetry-val" style="color: #f87171;">{} <span style="font-size: 1rem; color: #9ca3af;">kg</span></div>
                <div style="color: #9ca3af; font-size: 0.8rem; margin-top: 4px;">Est. Trip Carbon</div>
            </div>
            """.format(summary["total_co2_kg"]), unsafe_allow_html=True)

        with k5:
            delta_color = "#f87171" if summary["pct_diff_from_baseline"] > 0 else "#34d399"
            diff_sign = "+" if summary["pct_diff_from_baseline"] > 0 else ""
            st.markdown("""
            <div class="telemetry-card">
                <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">⚡ Route Avg Economy</div>
                <div class="telemetry-val" style="color: #a78bfa;">{} <span style="font-size: 1rem; color: #9ca3af;">L/100km</span></div>
                <div style="color: {}; font-size: 0.8rem; margin-top: 4px;">{}% vs base spec ({} MPG)</div>
            </div>
            """.format(
                summary["trip_avg_l100km"],
                delta_color,
                f"{diff_sign}{summary['pct_diff_from_baseline']}",
                summary["trip_avg_mpg"]
            ), unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Interactive Route Map with OpenStreetMap
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🗺️ Interactive GPS Route Map & Fuel Intensity Heatmap")
        
        map_c1, map_c2 = st.columns([3, 1])
        with map_c2:
            map_color_metric = st.radio(
                "🎨 Color Route Points By:",
                ["Fuel Consumption (L/100 km)", "Road Speed (km/h)", "Elevation (meters)", "Slope Gradient (%)"],
                index=0
            )
            color_column_map = {
                "Fuel Consumption (L/100 km)": "Fuel_Rate_L100km",
                "Road Speed (km/h)": "Speed_kmh",
                "Elevation (meters)": "Elevation_m",
                "Slope Gradient (%)": "Slope_pct"
            }
            selected_col = color_column_map[map_color_metric]
            
            st.markdown("##### 📍 Map Waypoint Highlights:")
            st.markdown(f"- 🟢 **Origin**: `{df_results['Latitude'].iloc[0]:.4f}, {df_results['Longitude'].iloc[0]:.4f}`")
            st.markdown(f"- 🏁 **Destination**: `{df_results['Latitude'].iloc[-1]:.4f}, {df_results['Longitude'].iloc[-1]:.4f}`")
            st.markdown(f"- ⛰️ **Max Elevation**: `{df_results['Elevation_m'].max():.1f} meters`")
            st.markdown(f"- 🚀 **Max Speed**: `{df_results['Speed_kmh'].max():.1f} km/h`")
            st.markdown(f"- ⛽ **Peak Fuel Burn**: `{df_results['Fuel_Rate_L100km'].max():.1f} L/100km`")

        with map_c1:
            # Create Plotly Map supporting both Plotly 7+ (scatter_map) and Plotly 5-6 (scatter_mapbox)
            is_v7 = hasattr(px, 'scatter_map')
            map_fn = px.scatter_map if is_v7 else px.scatter_mapbox
            scatter_trace = go.Scattermap if hasattr(go, 'Scattermap') else go.Scattermapbox
            style_kwargs = {'map_style': 'open-street-map'} if is_v7 else {'mapbox_style': 'carto-positron'}
            
            fig_map = map_fn(
                df_results,
                lat="Latitude",
                lon="Longitude",
                color=selected_col,
                size=[7] * len(df_results),
                size_max=10,
                color_continuous_scale="Turbo" if selected_col != "Fuel_Rate_L100km" else "Viridis",
                hover_name="Waypoint",
                hover_data={
                    "Distance_km": ':.1f',
                    "Speed_kmh": ':.1f',
                    "Elevation_m": ':.1f',
                    "Slope_pct": ':.2f',
                    "Fuel_Rate_L100km": ':.2f',
                    "Cumulative_Fuel_L": ':.2f',
                    "Latitude": False,
                    "Longitude": False
                },
                zoom=7 if summary["total_distance_km"] < 250 else 6,
                center=dict(lat=float(df_results["Latitude"].mean()), lon=float(df_results["Longitude"].mean())),
                **style_kwargs
            )

            # Add start and end pins
            fig_map.add_trace(scatter_trace(
                lat=[df_results['Latitude'].iloc[0]],
                lon=[df_results['Longitude'].iloc[0]],
                mode='markers+text',
                marker=dict(size=14, color='#10B981'),
                text=["🟢 START"],
                textposition="top right",
                name="Origin"
            ))

            fig_map.add_trace(scatter_trace(
                lat=[df_results['Latitude'].iloc[-1]],
                lon=[df_results['Longitude'].iloc[-1]],
                mode='markers+text',
                marker=dict(size=14, color='#EF4444'),
                text=["🏁 FINISH"],
                textposition="top right",
                name="Destination"
            ))

            fig_map.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f9fafb")
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Elevation Profile & Fuel Flow Telemetry Plots
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Route Elevation Profile & Instantaneous Fuel Telemetry")
        
        p_c1, p_c2 = st.columns(2)
        
        with p_c1:
            fig_elev = go.Figure()
            fig_elev.add_trace(go.Scatter(
                x=df_results['Distance_km'],
                y=df_results['Elevation_m'],
                mode='lines',
                fill='tozeroy',
                name='Elevation (m)',
                line=dict(color='#38bdf8', width=2.5),
                fillcolor='rgba(56, 189, 248, 0.2)'
            ))
            fig_elev.update_layout(
                title="🏔️ Terrain Elevation Profile (Meters above Sea Level)",
                xaxis_title="Distance along Route (km)",
                yaxis_title="Altitude (m)",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17, 24, 39, 0.5)",
                font=dict(color="#f9fafb"),
                margin=dict(l=30, r=20, t=40, b=30)
            )
            st.plotly_chart(fig_elev, use_container_width=True)
            
        with p_c2:
            fig_fuel_flow = go.Figure()
            fig_fuel_flow.add_trace(go.Scatter(
                x=df_results['Distance_km'],
                y=df_results['Fuel_Rate_L100km'],
                mode='lines',
                name='Instant Fuel Rate (L/100km)',
                line=dict(color='#f59e0b', width=2.2)
            ))
            fig_fuel_flow.add_trace(go.Scatter(
                x=df_results['Distance_km'],
                y=[pred_l100km] * len(df_results),
                mode='lines',
                name='Baseline ML Flat Rate',
                line=dict(color='#9ca3af', dash='dash', width=1.8)
            ))
            fig_fuel_flow.update_layout(
                title="⚡ Instantaneous Fuel Consumption Rate (L/100 km)",
                xaxis_title="Distance along Route (km)",
                yaxis_title="Fuel Rate (L/100km)",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17, 24, 39, 0.5)",
                font=dict(color="#f9fafb"),
                margin=dict(l=30, r=20, t=40, b=30)
            )
            st.plotly_chart(fig_fuel_flow, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Multi-Route Eco Optimizer (Alternative Routing Comparison)
        # ----------------------------------------------------------------------
        if not uploaded_mode:
            st.markdown("---")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🌿 Multi-Route Eco Optimizer: Route Options Compared")
            st.caption("AI analyzes route alternatives to balance driving duration, fuel economy, and carbon emissions.")
            
            eco_routes = generate_eco_route_alternatives(
                base_route_key=selected_route_key if selected_route_key in PRESET_ROUTES else list(PRESET_ROUTES.keys())[0],
                vehicle_specs=input_dict,
                base_l100km=pred_l100km,
                fuel_price=gps_fuel_price
            )
            
            ec1, ec2, ec3 = st.columns(3)
            
            for idx, (r_name, r_data) in enumerate(eco_routes.items()):
                target_col = [ec1, ec2, ec3][idx]
                card_class = "route-card route-recommended" if r_data.get("is_recommended") else "route-card"
                badge_text = "🏆 RECOMMENDED ECO CHOICE" if r_data.get("is_recommended") else r_data["tag"]
                
                with target_col:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">{r_name}</span>
                        </div>
                        <span style="background: {r_data['badge_color']}33; color: {r_data['badge_color']}; border: 1px solid {r_data['badge_color']}; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">
                            {badge_text}
                        </span>
                        <div style="margin-top: 1rem;">
                            <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff;">{r_data['fuel_liters']} <span style="font-size: 0.9rem; color: #9ca3af;">Liters</span></div>
                            <div style="color: #38bdf8; font-size: 0.9rem;">Cost: <b>${r_data['cost_usd']}</b> | Time: <b>{r_data['duration_hours']} hrs</b></div>
                            <div style="color: #9ca3af; font-size: 0.85rem; margin-top: 6px;">
                                • Distance: <b>{r_data['distance_km']} km</b><br>
                                • Fuel Rate: <b>{r_data['fuel_l100km']} L/100km</b><br>
                                • CO₂: <b>{r_data['co2_kg']} kg</b>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Route Comparison Bar Chart
            comp_df = pd.DataFrame([
                {
                    "Route": name.replace("🌿 ", "").replace("⚡ ", "").replace("🏙️ ", ""),
                    "Fuel Required (L)": d["fuel_liters"],
                    "Trip Cost ($)": d["cost_usd"],
                    "Travel Time (hrs)": d["duration_hours"],
                    "CO₂ Emission (kg)": d["co2_kg"]
                }
                for name, d in eco_routes.items()
            ])
            
            fig_comp = px.bar(
                comp_df,
                x="Route",
                y=["Fuel Required (L)", "Trip Cost ($)", "CO₂ Emission (kg)"],
                barmode="group",
                title="Route Performance Metrics Comparison",
                color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b"]
            )
            fig_comp.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f9fafb"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Live GPS Trip Simulator & Telemetry Replay
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏎️ Live GPS Trip Telemetry Simulator & Odometer Replay")
        st.caption("Drag the slider to simulate moving along the route and observe real-time telemetry, instant burn rates, and fuel tank depletion.")
        
        sim_point = st.slider(
            "🚗 Virtual Vehicle Position along Route (% Progress):",
            min_value=0,
            max_value=len(df_results) - 1,
            value=int(len(df_results) * 0.42),
            format="Waypoint %d"
        )
        
        current_wp = df_results.iloc[sim_point]
        tank_size_l = 55.0  # standard vehicle fuel tank
        fuel_remaining_l = max(0.0, tank_size_l - current_wp['Cumulative_Fuel_L'])
        tank_pct = (fuel_remaining_l / tank_size_l) * 100.0
        
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Current Odometer", f"{current_wp['Distance_km']:.1f} km", f"+{current_wp['Segment_Distance_km']:.2f} km/step")
        s2.metric("Speedometer", f"{current_wp['Speed_kmh']:.1f} km/h", f"Slope: {current_wp['Slope_pct']:+.1f}%")
        s3.metric("Instant Burn Rate", f"{current_wp['Fuel_Rate_L100km']:.2f} L/100km", "Live Flow")
        s4.metric("Altitude (MSL)", f"{current_wp['Elevation_m']:.0f} m", f"GPS WPT #{sim_point+1}")
        s5.metric("Fuel Tank Remaining", f"{fuel_remaining_l:.1f} L", f"{tank_pct:.1f}% Full")
        
        # Virtual Fuel Tank Progress Bar
        st.progress(int(np.clip(tank_pct, 0, 100)))
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 3: AUTOMOTIVE EDA & INSIGHTS
# ==============================================================================
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


# ==============================================================================
# TAB 4: MODEL BENCHMARKS & FEATURE IMPORTANCE
# ==============================================================================
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


# ==============================================================================
# TAB 5: ECO-DRIVING GUIDE
# ==============================================================================
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

    3. **GPS Road Topography & Grade Resistance ($\mathbf{F_g = m \cdot g \cdot \sin\theta}$)**
       - Climbing a 5% road incline increases instantaneous fuel consumption by up to 40-70%.
       - Coasting downhill and regenerative braking in Hybrids recaptures significant kinetic and potential energy.

    4. **Aerodynamic Drag ($\mathbf{F_d = \frac{1}{2} \rho C_d A v^2}$)**
       - At highway speeds ($> 80 \text{ km/h}$), aerodynamic drag dominates fuel consumption. Driving at 110 km/h consumes ~15-20% more fuel than at 90 km/h.

    ---

    #### 💡 Practical Eco-Driving Tips for Automotive Engineers & Drivers

    - **Proper Tire Inflation**: Low tire pressure increases rolling resistance coefficient ($C_{rr}$), increasing fuel usage by up to 3%.
    - **Maintain Smooth Acceleration**: Rapid acceleration operates the engine in rich fuel mixture zones, consuming extra fuel.
    - **Use GPS Eco-Routing**: Avoid steep hill climbs and congested traffic bottlenecks by selecting green bypass corridors.
    """)

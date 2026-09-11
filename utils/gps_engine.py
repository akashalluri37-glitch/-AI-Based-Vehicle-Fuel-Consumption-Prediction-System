# ==============================================================================
# Script: gps_engine.py
# Purpose: Advanced GPS Route Telemetry, Physical Fuel Consumption Modeling,
#          Elevation Gradient & Aerodynamic Analysis, and Eco-Routing Optimization.
# ==============================================================================

import numpy as np
import pandas as pd
import math

# Earth Radius in Kilometers for Haversine Distance
EARTH_RADIUS_KM = 6371.0

# ------------------------------------------------------------------------------
# 1. Geographic & Distance Utilities
# ------------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two GPS coordinates in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    
    return EARTH_RADIUS_KM * c

def interpolate_route_waypoints(start_lat, start_lon, end_lat, end_lon, n_points=50, curvature=0.015):
    """
    Generates realistic curving GPS waypoints along a road corridor between two coordinates.
    """
    lats = []
    lons = []
    
    # Perpendicular direction vector for realistic highway curve
    dlat = end_lat - start_lat
    dlon = end_lon - start_lon
    perp_lat = -dlon
    perp_lon = dlat
    norm = math.sqrt(perp_lat**2 + perp_lon**2) + 1e-9
    perp_lat /= norm
    perp_lon /= norm
    
    t_vals = np.linspace(0, 1, n_points)
    for i, t in enumerate(t_vals):
        # Base straight linear interpolation
        base_lat = start_lat + t * dlat
        base_lon = start_lon + t * dlon
        
        # Superimpose multi-frequency sinusoidal curves to mimic real road bends
        curve_offset = (
            math.sin(t * math.pi) * curvature * 1.0 +
            math.sin(t * 3 * math.pi) * (curvature * 0.4) +
            math.cos(t * 5 * math.pi) * (curvature * 0.2)
        )
        
        lat = base_lat + perp_lat * curve_offset
        lon = base_lon + perp_lon * curve_offset
        lats.append(round(lat, 6))
        lons.append(round(lon, 6))
        
    return lats, lons


# ------------------------------------------------------------------------------
# 2. Built-in Real-World GPS Route Catalog
# ------------------------------------------------------------------------------
PRESET_ROUTES = {
    "🏔️ San Francisco ➔ Lake Tahoe (Mountain Climb)": {
        "name": "San Francisco to Lake Tahoe",
        "description": "High-altitude mountain climb across the Sierra Nevada. High elevation gain (+1,900m), highway & steep alpine passes.",
        "start_city": "San Francisco, CA",
        "end_city": "Lake Tahoe, CA",
        "start_coords": (37.7749, -122.4194),
        "end_coords": (39.0968, -120.0324),
        "nominal_distance_km": 315.0,
        "base_elevation_start": 15.0,
        "base_elevation_end": 1900.0,
        "elevation_peak": 2200.0,
        "terrain_type": "Mountainous",
        "avg_speed_kmh": 88.0,
        "traffic_level": "Moderate"
    },
    "🚗 New York City ➔ Boston (I-95 Corridor)": {
        "name": "New York City to Boston",
        "description": "Interstate highway cruising along the Northeast corridor with coastal rolling hills and moderate urban bottlenecks.",
        "start_city": "New York, NY",
        "end_city": "Boston, MA",
        "start_coords": (40.7128, -74.0060),
        "end_coords": (42.3601, -71.0589),
        "nominal_distance_km": 345.0,
        "base_elevation_start": 10.0,
        "base_elevation_end": 40.0,
        "elevation_peak": 180.0,
        "terrain_type": "Rolling Hills",
        "avg_speed_kmh": 96.0,
        "traffic_level": "Heavy"
    },
    "🌊 Los Angeles ➔ San Diego (Pacific Coast I-5)": {
        "name": "Los Angeles to San Diego",
        "description": "Scenic coastal highway route along Southern California with stop-and-go city bypasses and smooth coastal cruising.",
        "start_city": "Los Angeles, CA",
        "end_city": "San Diego, CA",
        "start_coords": (34.0522, -118.2437),
        "end_coords": (32.7157, -117.1611),
        "nominal_distance_km": 195.0,
        "base_elevation_start": 85.0,
        "base_elevation_end": 20.0,
        "elevation_peak": 220.0,
        "terrain_type": "Coastal Flat & Low Hills",
        "avg_speed_kmh": 90.0,
        "traffic_level": "Heavy"
    },
    "🏰 London ➔ Oxford (M40 Motorway)": {
        "name": "London to Oxford",
        "description": "UK motorway and country bypass traversing gentle Chiltern Hills into Oxfordshire.",
        "start_city": "London, UK",
        "end_city": "Oxford, UK",
        "start_coords": (51.5074, -0.1278),
        "end_coords": (51.7520, -1.2577),
        "nominal_distance_km": 92.0,
        "base_elevation_start": 35.0,
        "base_elevation_end": 75.0,
        "elevation_peak": 240.0,
        "terrain_type": "Rolling Lowlands",
        "avg_speed_kmh": 92.0,
        "traffic_level": "Moderate"
    },
    "⛰️ Mumbai ➔ Pune (Western Ghats Expressway)": {
        "name": "Mumbai to Pune",
        "description": "Access-controlled expressway featuring the steep Bhor Ghat mountain incline with tunnel passes and fast plains.",
        "start_city": "Mumbai, India",
        "end_city": "Pune, India",
        "start_coords": (19.0760, 72.8777),
        "end_coords": (18.5204, 73.8567),
        "nominal_distance_km": 152.0,
        "base_elevation_start": 14.0,
        "base_elevation_end": 560.0,
        "elevation_peak": 680.0,
        "terrain_type": "Mountain Ghat / Expressway",
        "avg_speed_kmh": 82.0,
        "traffic_level": "Moderate"
    }
}


# ------------------------------------------------------------------------------
# 3. Route Generation & Telemetry Simulation
# ------------------------------------------------------------------------------
def generate_route_telemetry(route_key, custom_params=None, n_waypoints=80, seed=42):
    """
    Constructs a detailed GPS waypoint sequence with realistic:
    - Latitude & Longitude waypoints
    - Elevation profile and slope gradients (%)
    - Segment road speeds (km/h)
    - Traffic density and idle delay segments
    """
    np.random.seed(seed)
    
    if route_key in PRESET_ROUTES:
        info = PRESET_ROUTES[route_key]
        start_lat, start_lon = info["start_coords"]
        end_lat, end_lon = info["end_coords"]
        target_dist = info["nominal_distance_km"]
        elev_start = info["base_elevation_start"]
        elev_end = info["base_elevation_end"]
        elev_peak = info["elevation_peak"]
        base_speed = info["avg_speed_kmh"]
        terrain = info["terrain_type"]
    else:
        # Custom route specifications
        custom_params = custom_params or {}
        start_lat = custom_params.get("start_lat", 37.77)
        start_lon = custom_params.get("start_lon", -122.41)
        end_lat = custom_params.get("end_lat", 38.58)
        end_lon = custom_params.get("end_lon", -121.49)
        target_dist = custom_params.get("distance_km", 140.0)
        elev_start = custom_params.get("elev_start", 20.0)
        elev_end = custom_params.get("elev_end", 80.0)
        elev_peak = custom_params.get("elev_peak", 150.0)
        base_speed = custom_params.get("avg_speed", 90.0)
        terrain = custom_params.get("terrain", "Rolling Hills")

    # Generate GPS spatial coordinates
    lats, lons = interpolate_route_waypoints(start_lat, start_lon, end_lat, end_lon, n_points=n_waypoints)
    
    # Calculate step distances
    step_distances = [0.0]
    for i in range(1, len(lats)):
        d = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
        step_distances.append(d)
        
    raw_total_dist = sum(step_distances)
    scale_factor = target_dist / (raw_total_dist if raw_total_dist > 0 else 1.0)
    scaled_step_distances = [d * scale_factor for d in step_distances]
    cum_distances = np.cumsum(scaled_step_distances)
    
    # Generate Elevation Profile along route (meters above sea level)
    t = np.linspace(0, 1, n_waypoints)
    if "Mountain" in terrain:
        # Mountain climb with ascent, peak pass, and high plateau
        elev_profile = (
            elev_start + 
            (elev_end - elev_start) * (t ** 1.3) +
            (elev_peak - max(elev_start, elev_end)) * np.sin(np.pi * (t ** 0.8)) * 1.1 +
            np.random.normal(0, 8, n_waypoints)
        )
    elif "Rolling" in terrain or "Hills" in terrain:
        elev_profile = (
            elev_start + (elev_end - elev_start) * t +
            (elev_peak - min(elev_start, elev_end)) * 0.4 * np.sin(2 * np.pi * t) +
            np.sin(6 * np.pi * t) * 15.0 +
            np.random.normal(0, 4, n_waypoints)
        )
    else: # Coastal / Flat
        elev_profile = (
            elev_start + (elev_end - elev_start) * t +
            np.sin(np.pi * t) * 20.0 +
            np.random.normal(0, 2, n_waypoints)
        )
    elev_profile = np.clip(elev_profile, 0, 4000)

    # Compute Slope Gradient (%) = (Delta Altitude / Delta Distance) * 100
    slopes = [0.0]
    for i in range(1, n_waypoints):
        delta_h_m = elev_profile[i] - elev_profile[i-1]
        delta_d_m = scaled_step_distances[i] * 1000.0  # convert km to meters
        slope_pct = (delta_h_m / max(delta_d_m, 1.0)) * 100.0
        # Realistic road slope limits (-15% to +15%)
        slopes.append(float(np.clip(slope_pct, -15.0, 15.0)))

    # Compute Speed Profile (km/h) across route segments
    speeds = []
    for i, s in enumerate(slopes):
        progress = t[i]
        # City exit and city entrance are slower
        if progress < 0.08 or progress > 0.92:
            seg_base = base_speed * 0.45 + np.random.normal(0, 5)
        # Mid-route highway cruising
        else:
            seg_base = base_speed + np.random.normal(0, 6)
            
        # Slope impact on speed: uphill vehicles slow slightly, downhill faster
        if s > 3.0:
            seg_base -= (s * 1.8)
        elif s < -3.0:
            seg_base += 4.0
            
        speeds.append(float(np.clip(seg_base, 20.0, 130.0)))

    # Traffic Congestion Index (0 = Free Flow, 1 = Gridlock / Heavy Stop & Go)
    traffic_density = []
    for i in range(n_waypoints):
        prog = t[i]
        # High traffic at origin & destination urban centers, plus occasional mid-route bottlenecks
        if prog < 0.12 or prog > 0.88:
            density = np.clip(np.random.uniform(0.5, 0.9), 0, 1)
        elif 0.45 <= prog <= 0.55 and np.random.rand() > 0.4:
            density = np.clip(np.random.uniform(0.3, 0.7), 0, 1)
        else:
            density = np.clip(np.random.uniform(0.05, 0.25), 0, 1)
        traffic_density.append(float(density))

    # Compile into structured DataFrame
    df_telemetry = pd.DataFrame({
        'Waypoint': list(range(1, n_waypoints + 1)),
        'Latitude': lats,
        'Longitude': lons,
        'Distance_km': np.round(cum_distances, 2),
        'Segment_Distance_km': np.round(scaled_step_distances, 3),
        'Elevation_m': np.round(elev_profile, 1),
        'Slope_pct': np.round(slopes, 2),
        'Speed_kmh': np.round(speeds, 1),
        'Traffic_Density': np.round(traffic_density, 2)
    })
    
    return df_telemetry


# ------------------------------------------------------------------------------
# 4. Physical GPS Fuel Consumption Engine
# ------------------------------------------------------------------------------
def predict_gps_route_fuel(df_telemetry, vehicle_specs, base_model_l100km, driving_style="🚗 Normal Balanced", fuel_price_per_l=1.60):
    r"""
    Calculates detailed physical fuel consumption for each segment of a GPS route.
    
    Factors considered:
    1. Base Machine Learning Fuel Rate (L/100km)
    2. Vehicle Mass & Gravitational Potential Energy ($F_g = m \cdot g \cdot \sin\theta$)
    3. Aerodynamic Drag ($F_d = \frac{1}{2} \rho C_d A v^2$)
    4. Powertrain Type (Petrol, Diesel, Hybrid regenerative recovery, PHEV)
    5. Traffic Idling & Acceleration dynamics
    6. Driver style multiplier (Eco / Normal / Aggressive)
    """
    weight_kg = vehicle_specs.get('Vehicle Weight', 1500)
    fuel_type = vehicle_specs.get('Fuel Type', 'Petrol')
    transmission = vehicle_specs.get('Transmission', 'Automatic')
    horsepower = vehicle_specs.get('Horsepower', 160)
    engine_size = vehicle_specs.get('Engine Size', 2.0)
    
    # Driver style acceleration penalty
    style_factors = {
        "🌿 Eco-Conscious": 0.88,
        "🚗 Normal Balanced": 1.00,
        "🏎️ Dynamic / Aggressive": 1.22
    }
    style_mult = style_factors.get(driving_style, 1.00)

    # Hybrid regenerative braking recovery efficiency
    is_hybrid = fuel_type in ['Hybrid', 'Plug-in Hybrid']
    regen_factor = 0.45 if is_hybrid else 0.08  # Hybrids reclaim downhill kinetic energy

    # Segment calculations
    segment_fuel_liters = []
    segment_rates_l100km = []
    instant_co2_kg = []
    segment_times_min = []
    
    cum_fuel = 0.0
    cum_fuel_list = []
    
    # CO2 emission factor per liter
    co2_per_liter = 2.31 if fuel_type != 'Diesel' else 2.68

    for _, row in df_telemetry.iterrows():
        d_km = row['Segment_Distance_km']
        v_kmh = row['Speed_kmh']
        slope = row['Slope_pct']
        traffic = row['Traffic_Density']
        
        if d_km <= 0.0001:
            segment_fuel_liters.append(0.0)
            segment_rates_l100km.append(base_model_l100km)
            instant_co2_kg.append(0.0)
            segment_times_min.append(0.0)
            cum_fuel_list.append(0.0)
            continue
            
        # Segment Travel Time (hours & minutes)
        t_hours = d_km / max(v_kmh, 5.0)
        t_minutes = t_hours * 60.0
        segment_times_min.append(round(t_minutes, 2))
        
        # 1. Aerodynamic Speed Factor:
        # Optimal cruising speed is ~75 km/h. At >100 km/h, drag grows with v^2.
        # At very low speeds (<30 km/h), engine thermal efficiency drops.
        if v_kmh > 75.0:
            speed_mult = 1.0 + 0.000085 * ((v_kmh - 75.0) ** 2)
        elif v_kmh < 45.0:
            speed_mult = 1.0 + 0.006 * (45.0 - v_kmh)
        else:
            speed_mult = 0.95  # Sweet spot cruising efficiency
            
        # 2. Elevation & Slope Factor:
        # Climbing requires mechanical energy: P = m * g * v * sin(theta)
        # Normalized against baseline vehicle weight
        weight_norm = weight_kg / 1500.0
        if slope > 0:
            # Uphill penalty: +6% to +14% fuel consumption per 1% gradient
            slope_mult = 1.0 + (slope * 0.085 * weight_norm)
        else:
            # Downhill relief: reduced throttle, coasting, and regenerative recovery
            slope_abs = abs(slope)
            relief = slope_abs * 0.065 * (1.0 + regen_factor)
            slope_mult = max(0.18, 1.0 - relief)
            
        # 3. Traffic Congestion & Stop-and-Go Penalty:
        # Engine idle burn (~0.8 L/hr for 2.0L engine) + Acceleration inertia restarts
        idle_burn_rate_l_hr = 0.4 + (engine_size * 0.25)
        if is_hybrid:
            idle_burn_rate_l_hr *= 0.20  # Hybrids turn off engine when idling
            
        idle_fuel_liters = traffic * t_hours * idle_burn_rate_l_hr
        traffic_mult = 1.0 + (traffic * 0.35)

        # Combined Instantaneous Fuel Consumption Rate (L/100 km)
        instant_rate = base_model_l100km * speed_mult * slope_mult * traffic_mult * style_mult
        instant_rate = float(np.clip(instant_rate, 1.8, 32.0))
        
        # Segment Fuel in Liters
        seg_fuel = (instant_rate / 100.0) * d_km + idle_fuel_liters
        seg_fuel = round(float(seg_fuel), 4)
        
        cum_fuel += seg_fuel
        
        segment_fuel_liters.append(seg_fuel)
        segment_rates_l100km.append(round(instant_rate, 2))
        instant_co2_kg.append(round(seg_fuel * co2_per_liter, 3))
        cum_fuel_list.append(round(cum_fuel, 3))

    # Add calculated columns to telemetry DataFrame
    df_results = df_telemetry.copy()
    df_results['Segment_Time_min'] = segment_times_min
    df_results['Segment_Fuel_L'] = segment_fuel_liters
    df_results['Cumulative_Fuel_L'] = cum_fuel_list
    df_results['Fuel_Rate_L100km'] = segment_rates_l100km
    df_results['Segment_CO2_kg'] = instant_co2_kg

    # Overall Trip Aggregates
    total_distance_km = float(df_results['Distance_km'].iloc[-1])
    total_fuel_liters = round(float(cum_fuel), 2)
    total_cost_usd = round(total_fuel_liters * fuel_price_per_l, 2)
    total_co2_kg = round(total_fuel_liters * co2_per_liter, 2)
    total_time_hours = round(df_results['Segment_Time_min'].sum() / 60.0, 2)
    
    trip_avg_l100km = round((total_fuel_liters / max(total_distance_km, 1.0)) * 100.0, 2)
    trip_avg_mpg = round(235.215 / trip_avg_l100km, 1) if trip_avg_l100km > 0 else 0.0

    # Fuel Economy Rating vs Baseline ML prediction
    pct_diff = round(((trip_avg_l100km - base_model_l100km) / base_model_l100km) * 100.0, 1)
    
    # Efficiency Score (0 - 100)
    ideal_fuel = (base_model_l100km * 0.90 / 100.0) * total_distance_km
    efficiency_score = int(np.clip(100 - max(0, (total_fuel_liters - ideal_fuel) / ideal_fuel * 60), 30, 99))

    summary = {
        "total_distance_km": total_distance_km,
        "total_distance_miles": round(total_distance_km * 0.621371, 1),
        "total_fuel_liters": total_fuel_liters,
        "total_fuel_gallons": round(total_fuel_liters * 0.264172, 2),
        "total_cost_usd": total_cost_usd,
        "total_co2_kg": total_co2_kg,
        "total_time_hours": total_time_hours,
        "trip_avg_l100km": trip_avg_l100km,
        "trip_avg_mpg": trip_avg_mpg,
        "base_model_l100km": base_model_l100km,
        "pct_diff_from_baseline": pct_diff,
        "efficiency_score": efficiency_score,
        "fuel_price_per_l": fuel_price_per_l
    }

    return df_results, summary


# ------------------------------------------------------------------------------
# 5. Multi-Route Eco Optimizer (Alternative Route Comparison)
# ------------------------------------------------------------------------------
def generate_eco_route_alternatives(base_route_key, vehicle_specs, base_l100km, fuel_price=1.60):
    """
    Generates and compares 3 alternative routes for a trip:
    1. ⚡ Fast Highway Route (High speed, fast time, moderate-high fuel)
    2. 🌿 Eco-Friendly Green Route (Optimal 75-80 km/h speed, flat elevation, lowest fuel & CO2)
    3. 🏙️ Shortest Urban Route (Shortest distance, but heavy traffic signals, stop-and-go)
    """
    base_info = PRESET_ROUTES.get(base_route_key, list(PRESET_ROUTES.values())[0])
    dist = base_info["nominal_distance_km"]
    
    # 1. Fast Highway
    highway_dist = dist * 1.03
    highway_speed = 110.0
    highway_elev_penalty = 1.02
    highway_speed_mult = 1.18 # High aerodynamic drag
    highway_l100km = base_l100km * highway_speed_mult * highway_elev_penalty
    highway_fuel = round((highway_l100km / 100.0) * highway_dist, 2)
    highway_time = round(highway_dist / highway_speed, 2)
    highway_cost = round(highway_fuel * fuel_price, 2)
    highway_co2 = round(highway_fuel * 2.31, 2)
    
    # 2. Eco Green Route
    eco_dist = dist * 1.01
    eco_speed = 78.0 # Optimal cruising window
    eco_l100km = base_l100km * 0.91 # Optimal aerodynamic and kinetic zone
    eco_fuel = round((eco_l100km / 100.0) * eco_dist, 2)
    eco_time = round(eco_dist / eco_speed, 2)
    eco_cost = round(eco_fuel * fuel_price, 2)
    eco_co2 = round(eco_fuel * 2.31, 2)
    
    # 3. Shortest Urban / Direct Route
    urban_dist = dist * 0.93 # 7% shorter distance
    urban_speed = 48.0 # Slower urban / arterial average
    urban_l100km = base_l100km * 1.28 # Stop & go acceleration penalties
    urban_fuel = round((urban_l100km / 100.0) * urban_dist, 2)
    urban_time = round(urban_dist / urban_speed, 2)
    urban_cost = round(urban_fuel * fuel_price, 2)
    urban_co2 = round(urban_fuel * 2.31, 2)
    
    # Fuel and Money Saved by picking Eco Route vs Highway & Urban
    fuel_saved_vs_highway = round(highway_fuel - eco_fuel, 2)
    cost_saved_vs_highway = round(highway_cost - eco_cost, 2)
    co2_saved_vs_highway = round(highway_co2 - eco_co2, 2)
    
    cost_saved_vs_urban = round(urban_cost - eco_cost, 2)

    routes = {
        "🌿 Eco-Friendly Green Route": {
            "icon": "🌿",
            "tag": "Lowest Fuel & Carbon",
            "distance_km": round(eco_dist, 1),
            "duration_hours": eco_time,
            "avg_speed_kmh": eco_speed,
            "fuel_liters": eco_fuel,
            "fuel_l100km": round(eco_l100km, 2),
            "cost_usd": eco_cost,
            "co2_kg": eco_co2,
            "fuel_saving_liters": fuel_saved_vs_highway,
            "cost_saving_usd": cost_saved_vs_highway,
            "is_recommended": True,
            "badge_color": "#10B981"
        },
        "⚡ Fast Highway Route": {
            "icon": "⚡",
            "tag": "Fastest Arrival Time",
            "distance_km": round(highway_dist, 1),
            "duration_hours": highway_time,
            "avg_speed_kmh": highway_speed,
            "fuel_liters": highway_fuel,
            "fuel_l100km": round(highway_l100km, 2),
            "cost_usd": highway_cost,
            "co2_kg": highway_co2,
            "time_diff_min": round((eco_time - highway_time) * 60, 0),
            "is_recommended": False,
            "badge_color": "#3B82F6"
        },
        "🏙️ Shortest Urban Route": {
            "icon": "🏙️",
            "tag": "Minimum Distance",
            "distance_km": round(urban_dist, 1),
            "duration_hours": urban_time,
            "avg_speed_kmh": urban_speed,
            "fuel_liters": urban_fuel,
            "fuel_l100km": round(urban_l100km, 2),
            "cost_usd": urban_cost,
            "co2_kg": urban_co2,
            "fuel_saving_liters": 0.0,
            "is_recommended": False,
            "badge_color": "#F59E0B"
        }
    }
    
    return routes


# ------------------------------------------------------------------------------
# 6. Custom GPS File Parser (CSV & GPX Telemetry Ingestion)
# ------------------------------------------------------------------------------
def parse_custom_gps_csv(file_buffer):
    """
    Parses a user-uploaded CSV containing GPS coordinates & telemetry.
    Expected columns (flexible aliases):
    - Latitude: ['lat', 'latitude', 'y']
    - Longitude: ['lon', 'long', 'longitude', 'x']
    - Elevation: ['elevation', 'elev', 'altitude', 'alt', 'ele'] (Optional)
    - Speed: ['speed', 'speed_kmh', 'velocity'] (Optional)
    """
    df_raw = pd.read_csv(file_buffer)
    cols = {c.lower().strip(): c for c in df_raw.columns}
    
    # Identify Latitude
    lat_col = None
    for cand in ['lat', 'latitude', 'y']:
        if cand in cols:
            lat_col = cols[cand]
            break
            
    # Identify Longitude
    lon_col = None
    for cand in ['lon', 'long', 'longitude', 'x']:
        if cand in cols:
            lon_col = cols[cand]
            break
            
    if not lat_col or not lon_col:
        raise ValueError("CSV must contain valid 'latitude' (or 'lat') and 'longitude' (or 'lon') columns.")
        
    df_clean = pd.DataFrame()
    df_clean['Latitude'] = df_raw[lat_col].astype(float)
    df_clean['Longitude'] = df_raw[lon_col].astype(float)
    
    # Elevation column
    elev_col = None
    for cand in ['elevation', 'elev', 'altitude', 'alt', 'ele']:
        if cand in cols:
            elev_col = cols[cand]
            break
    if elev_col:
        df_clean['Elevation_m'] = df_raw[elev_col].astype(float)
    else:
        df_clean['Elevation_m'] = 50.0  # Default nominal altitude
        
    # Speed column
    speed_col = None
    for cand in ['speed', 'speed_kmh', 'velocity']:
        if cand in cols:
            speed_col = cols[cand]
            break
    if speed_col:
        df_clean['Speed_kmh'] = df_raw[speed_col].astype(float)
    else:
        df_clean['Speed_kmh'] = 80.0 # Default speed
        
    # Compute step distances
    step_dists = [0.0]
    for i in range(1, len(df_clean)):
        d = haversine_distance(
            df_clean['Latitude'].iloc[i-1], df_clean['Longitude'].iloc[i-1],
            df_clean['Latitude'].iloc[i], df_clean['Longitude'].iloc[i]
        )
        step_dists.append(d)
        
    df_clean['Segment_Distance_km'] = np.round(step_dists, 3)
    df_clean['Distance_km'] = np.round(np.cumsum(step_dists), 2)
    df_clean['Waypoint'] = list(range(1, len(df_clean) + 1))
    
    # Compute slope
    slopes = [0.0]
    for i in range(1, len(df_clean)):
        dh = df_clean['Elevation_m'].iloc[i] - df_clean['Elevation_m'].iloc[i-1]
        dd = df_clean['Segment_Distance_km'].iloc[i] * 1000.0
        s = (dh / max(dd, 1.0)) * 100.0
        slopes.append(float(np.clip(s, -15.0, 15.0)))
    df_clean['Slope_pct'] = np.round(slopes, 2)
    
    # Traffic default
    df_clean['Traffic_Density'] = 0.20
    
    return df_clean

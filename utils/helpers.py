import os
import json
import joblib

def load_models_and_artifacts():
    """
    Loads preprocessor, trained regression models, and evaluation metrics summary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models", "saved_models")
    
    artifacts = {
        "regressors": {
            "Linear Regression": joblib.load(os.path.join(models_dir, "regressor_linear_regression.joblib")),
            "Decision Tree": joblib.load(os.path.join(models_dir, "regressor_decision_tree.joblib")),
            "Random Forest": joblib.load(os.path.join(models_dir, "regressor_random_forest.joblib"))
        },
        "preprocessor": joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    }
    
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
    else:
        artifacts["metrics"] = {}
        
    return artifacts

def l100km_to_mpg(l100km):
    """
    Converts Liters per 100km to US Miles Per Gallon (MPG).
    """
    if l100km <= 0:
        return 0.0
    return round(235.215 / l100km, 1)

def get_efficiency_category(l100km):
    """
    Categorizes fuel consumption into engineering efficiency ratings.
    """
    mpg = l100km_to_mpg(l100km)
    
    if l100km < 6.0:
        return {
            "category": "High Efficiency (Eco Leader)",
            "badge_class": "badge-eco",
            "color": "#10B981",
            "icon": "🟢",
            "summary": f"Outstanding fuel economy ({l100km} L/100km = {mpg} MPG). Low carbon footprint."
        }
    elif l100km <= 9.0:
        return {
            "category": "Moderate Efficiency (Standard)",
            "badge_class": "badge-moderate",
            "color": "#F59E0B",
            "icon": "🟡",
            "summary": f"Typical average fuel consumption ({l100km} L/100km = {mpg} MPG) for commuter vehicles."
        }
    else:
        return {
            "category": "Low Efficiency (High Fuel Consumer)",
            "badge_class": "badge-heavy",
            "color": "#EF4444",
            "icon": "🔴",
            "summary": f"High fuel consumption ({l100km} L/100km = {mpg} MPG). Typical of heavy SUVs or sports cars."
        }

def calculate_annual_fuel_cost(l100km, annual_km=15000, fuel_price_per_l=1.50):
    """
    Calculates estimated annual fuel expenditure and CO2 emissions.
    """
    total_liters = (l100km / 100.0) * annual_km
    annual_cost = total_liters * fuel_price_per_l
    annual_co2_kg = total_liters * 2.31  # Approx 2.31 kg CO2 per liter of fuel
    
    return {
        "total_liters": round(total_liters, 1),
        "annual_cost": round(annual_cost, 2),
        "annual_co2_kg": round(annual_co2_kg, 1),
        "annual_co2_tons": round(annual_co2_kg / 1000.0, 2)
    }

def get_custom_css():
    """
    Returns modern Glassmorphism custom styling for Streamlit.
    """
    return """
    <style>
    /* Global App Container */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #1f2937 100%);
        color: #f9fafb;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px -5px rgba(37, 99, 235, 0.3);
        margin-bottom: 2rem;
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(31, 41, 55, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.2rem;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.15rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .badge-eco {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }
    
    .badge-moderate {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }
    
    .badge-heavy {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(17, 24, 39, 0.7);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }
    </style>
    """

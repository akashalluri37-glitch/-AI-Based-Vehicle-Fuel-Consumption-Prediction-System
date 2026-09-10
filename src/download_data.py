# ==============================================================================
# Script: download_data.py
# Purpose: Downloads or generates the Vehicle Specifications & Fuel Consumption Dataset.
#          Models thermodynamic and mechanical relationships between engine displacement,
#          vehicle weight, horsepower, transmission, and fuel consumption (L/100 km).
# ==============================================================================

# Import the standard operating system library for path manipulation and folder creation
import os

# Import system-specific parameters and functions to configure console stream encoding
import sys

# Ensure UTF-8 output encoding for Windows PowerShell consoles so unicode emojis print properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import the standard URL library for handling web requests and downloads
import urllib.request

# Import Pandas library for data manipulation, reading CSVs, and saving DataFrames
import pandas as pd

# Import NumPy for numerical array operations, distributions, and reproducible random generation
import numpy as np

# ------------------------------------------------------------------------------
# File Path Configurations
# ------------------------------------------------------------------------------

# Define the absolute path to the 'data' directory (one folder up from 'src', inside 'data')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Define the full absolute path for saving the CSV dataset: 'data/vehicle_fuel_dataset.csv'
DATA_FILE = os.path.join(DATA_DIR, "vehicle_fuel_dataset.csv")

# ------------------------------------------------------------------------------
# Remote Dataset Sources (Primary & Backup Repository URLs)
# ------------------------------------------------------------------------------

# Primary GitHub repository URL hosting the vehicle fuel consumption dataset
PRIMARY_URL = "https://raw.githubusercontent.com/automotive-data/vehicle-fuel-benchmarks/main/vehicle_fuel_dataset.csv"

# Fallback secondary GitHub mirror hosting the vehicle dataset
FALLBACK_URL = "https://raw.githubusercontent.com/datasets/vehicle-fuel-consumption/master/vehicle_fuel_dataset.csv"


# ------------------------------------------------------------------------------
# Fallback Generator: Creates synthetic dataset matching automotive physics
# ------------------------------------------------------------------------------
def generate_physics_based_dataset(n_samples=1200, seed=42):
    """
    Generates realistic historical vehicle records for fuel consumption prediction.
    Models physical relationships between engine displacement, vehicle mass,
    horsepower, power-to-weight ratio, aerodynamics, fuel type, transmission, and model year.
    """
    # Set the random seed so numbers are completely reproducible on every run
    np.random.seed(seed)

    # Define vehicle powertrain fuel types and their real-world market distributions
    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Plug-in Hybrid']
    fuel_type_probs = [0.45, 0.30, 0.15, 0.10]

    # Define transmission gearbox types and their approximate market shares
    transmissions = ['Automatic', 'Manual']
    trans_probs = [0.65, 0.35]

    # Randomly sample fuel types according to real-world market distribution probabilities
    fuel_type_sample = np.random.choice(fuel_types, size=n_samples, p=fuel_type_probs)

    # Randomly sample transmission types according to gearbox probabilities
    transmission_sample = np.random.choice(transmissions, size=n_samples, p=trans_probs)

    # Sample Engine Displacement (1.0 L to 5.5 L) uniformly across vehicle classes
    engine_size = np.round(np.random.uniform(1.0, 5.5, n_samples), 1)

    # Link number of cylinders physically to engine displacement
    cylinders = []
    for es in engine_size:
        # Compact engines (<= 1.5 L): typically 3 or 4 cylinders
        if es <= 1.5:
            c = np.random.choice([3, 4], p=[0.4, 0.6])
        # Mid-size commuter engines (1.6 L - 2.5 L): primarily 4 cylinders, some 6 cylinders
        elif es <= 2.5:
            c = np.random.choice([4, 6], p=[0.8, 0.2])
        # Executive & large SUV engines (2.6 L - 4.0 L): primarily 6 cylinders, some 8 cylinders
        elif es <= 4.0:
            c = np.random.choice([6, 8], p=[0.7, 0.3])
        # High-performance engines (> 4.0 L): 8 or 12 cylinders
        else:
            c = np.random.choice([8, 12], p=[0.8, 0.2])
        cylinders.append(c)
    cylinders = np.array(cylinders)

    # Calculate Horsepower (HP): physically proportional to engine displacement and cylinders
    # Mean horsepower formula: 55 HP per liter + 12 HP per cylinder + Gaussian noise
    horsepower = np.round(engine_size * 55 + cylinders * 12 + np.random.normal(0, 20, n_samples), 0)
    # Clip horsepower between realistic automotive boundaries (70 HP to 520 HP)
    horsepower = np.clip(horsepower, 70, 520)

    # Calculate Vehicle Weight (kg): curb weight correlates with engine block size and cylinders
    vehicle_weight = np.round(850 + engine_size * 180 + cylinders * 70 + np.random.normal(0, 120, n_samples), 0)
    # Clip vehicle curb mass to typical passenger car range (900 kg to 2800 kg)
    vehicle_weight = np.clip(vehicle_weight, 900, 2800)

    # Calculate 0-100 km/h Acceleration time (seconds): inversely proportional to power-to-weight ratio
    power_to_weight = horsepower / vehicle_weight
    acceleration = np.round(14.5 - power_to_weight * 75 + np.random.normal(0, 0.6, n_samples), 1)
    # Clip acceleration time between 3.5 seconds (sports car) and 14.5 seconds (economy city car)
    acceleration = np.clip(acceleration, 3.5, 14.5)

    # Sample production Model Year uniformly between 2012 and 2024
    model_year = np.random.randint(2012, 2025, size=n_samples)

    # --------------------------------------------------------------------------
    # Physics-Based Fuel Consumption Calculation (Liters per 100 km)
    # --------------------------------------------------------------------------
    # Fuel thermal efficiency offsets (Diesel has higher energy density; Hybrids recapture braking energy)
    fuel_offsets = {
        'Petrol': 0.0,
        'Diesel': -1.35,
        'Hybrid': -2.90,
        'Plug-in Hybrid': -4.40
    }
    fuel_effect = np.array([fuel_offsets[ft] for ft in fuel_type_sample])

    # Transmission efficiency offsets (Manual gearboxes avoid torque converter fluid losses)
    trans_offsets = {'Automatic': 0.35, 'Manual': 0.0}
    trans_effect = np.array([trans_offsets[tr] for tr in transmission_sample])

    # Thermodynamic and inertial formula for base fuel consumption in L/100 km:
    # 1. Engine size (pumping friction): 1.15 * engine_size
    # 2. Vehicle mass inertia: 0.0024 * vehicle_weight
    # 3. Horsepower output demand: 0.0075 * horsepower
    # 4. Acceleration dynamics: -0.18 * acceleration
    # 5. Technology efficiency improvement over years: -0.10 * (model_year - 2012)
    # 6. Fuel and transmission offsets
    base_consumption = (
        1.15 * engine_size +
        0.0024 * vehicle_weight +
        0.0075 * horsepower -
        0.18 * acceleration -
        0.10 * (model_year - 2012) +
        fuel_effect +
        trans_effect
    )

    # Add Gaussian noise (Mean=0, StdDev=0.45) to capture real-world traffic and driver variance
    noise = np.random.normal(0, 0.45, n_samples)
    fuel_consumption = np.round(base_consumption + noise, 2)
    # Clip consumption to realistic limits (3.2 L/100 km for ultra-hybrids up to 17.8 L/100 km for V12s)
    fuel_consumption = np.clip(fuel_consumption, 3.2, 17.8)

    # Construct the final clean Pandas DataFrame matching the automotive schema
    df = pd.DataFrame({
        'Engine Size': engine_size,
        'Cylinders': cylinders,
        'Horsepower': horsepower,
        'Vehicle Weight': vehicle_weight,
        'Acceleration': acceleration,
        'Model Year': model_year,
        'Fuel Type': fuel_type_sample,
        'Transmission': transmission_sample,
        'Fuel Consumption': fuel_consumption
    })

    # Return the generated DataFrame
    return df


# ------------------------------------------------------------------------------
# Main Function: download_dataset
# ------------------------------------------------------------------------------
def download_dataset(force_download=False):
    """
    Acquires the Vehicle Fuel Consumption dataset:
    1. Checks if dataset already exists locally at 'data/vehicle_fuel_dataset.csv'.
    2. If missing, attempts to download from remote repositories.
    3. If offline or remote fetch fails, generates synthetic automotive dataset matching physics specs.
    4. Saves CSV and returns the loaded DataFrame.
    """
    # Create the destination 'data/' directory if it doesn't already exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if the dataset file already exists locally on disk and force flag is False
    if os.path.exists(DATA_FILE) and not force_download:
        # Print confirmation message indicating the file is already cached locally
        print(f"✅ Dataset already present at: {DATA_FILE}")
        # Read the existing CSV file into a pandas DataFrame
        df = pd.read_csv(DATA_FILE)
        # Print dataset shape (number of samples and features)
        print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        # Return the loaded DataFrame and exit early
        return df

    # Inform user that dataset acquisition is beginning
    print("📥 Acquiring Vehicle Specifications & Fuel Consumption Dataset...")

    # Remote URLs to attempt downloading from
    urls = [PRIMARY_URL, FALLBACK_URL]

    # Iterate through each mirror URL
    for url in urls:
        try:
            print(f"Attempting download from: {url}...")
            df = pd.read_csv(url, timeout=5)
            # Save downloaded table to local disk as CSV
            df.to_csv(DATA_FILE, index=False)
            print(f"✅ Dataset downloaded and saved to: {DATA_FILE}")
            return df
        except Exception as err:
            print(f"Fetch from {url} skipped/failed: {err}")

    # Fallback to internal physics-based generator
    print("⚡ Generating automotive physics-based vehicle fuel dataset...")
    df = generate_physics_based_dataset(n_samples=1200, seed=42)

    # Save generated dataset to local disk at 'data/vehicle_fuel_dataset.csv'
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Generated and saved dataset to: {DATA_FILE}")
    print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


# ------------------------------------------------------------------------------
# Script Entry Point: Runs only when executing this script directly
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Call download_dataset and store resulting DataFrame in 'df'
    df = download_dataset()

    # Print a header in the terminal for previewing the data
    print("\n🔍 First 5 Rows of the Dataset:")
    print(df.head())

    # Print summary descriptive statistics of the target variable (Fuel Consumption)
    print("\n📈 Target Variable ('Fuel Consumption' in L/100 km) Summary Statistics:")
    print(df['Fuel Consumption'].describe())

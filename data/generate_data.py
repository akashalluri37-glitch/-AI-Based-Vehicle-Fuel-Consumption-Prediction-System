import numpy as np
import pandas as pd
import os

def generate_vehicle_dataset(n_samples=1200, seed=42):
    """
    Generates realistic historical vehicle dataset for fuel consumption prediction.
    Models physical relationships between engine displacement, vehicle mass,
    horsepower, aerodynamics/acceleration, fuel type, transmission, and model year.
    """
    np.random.seed(seed)
    
    # 1. Feature Specifications
    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Plug-in Hybrid']
    fuel_type_probs = [0.45, 0.30, 0.15, 0.10]
    
    transmissions = ['Automatic', 'Manual']
    trans_probs = [0.65, 0.35]
    
    # Sample Categorical Features
    fuel_type_sample = np.random.choice(fuel_types, size=n_samples, p=fuel_type_probs)
    transmission_sample = np.random.choice(transmissions, size=n_samples, p=trans_probs)
    
    # Sample Numerical Features
    engine_size = np.round(np.random.uniform(1.0, 5.5, n_samples), 1)
    
    # Cylinders linked roughly to engine size
    cylinders = []
    for es in engine_size:
        if es <= 1.5:
            c = np.random.choice([3, 4], p=[0.4, 0.6])
        elif es <= 2.5:
            c = np.random.choice([4, 6], p=[0.8, 0.2])
        elif es <= 4.0:
            c = np.random.choice([6, 8], p=[0.7, 0.3])
        else:
            c = np.random.choice([8, 12], p=[0.8, 0.2])
        cylinders.append(c)
    cylinders = np.array(cylinders)
    
    # Horsepower linked to engine size & cylinders
    horsepower = np.round(engine_size * 55 + cylinders * 12 + np.random.normal(0, 20, n_samples), 0)
    horsepower = np.clip(horsepower, 70, 520)
    
    # Vehicle Weight (kg) linked to engine size & cylinders
    vehicle_weight = np.round(850 + engine_size * 180 + cylinders * 70 + np.random.normal(0, 120, n_samples), 0)
    vehicle_weight = np.clip(vehicle_weight, 900, 2800)
    
    # Acceleration 0-100 km/h (sec) inversely proportional to power-to-weight ratio
    power_to_weight = horsepower / vehicle_weight
    acceleration = np.round(14.5 - power_to_weight * 75 + np.random.normal(0, 0.6, n_samples), 1)
    acceleration = np.clip(acceleration, 3.5, 14.5)
    
    # Model Year (2012 to 2024)
    model_year = np.random.randint(2012, 2025, size=n_samples)
    
    # 2. Physics-Based Fuel Consumption Calculation (L/100 km)
    # Offsets for Fuel Type
    fuel_offsets = {
        'Petrol': 0.0,
        'Diesel': -1.35,
        'Hybrid': -2.90,
        'Plug-in Hybrid': -4.40
    }
    fuel_effect = np.array([fuel_offsets[ft] for ft in fuel_type_sample])
    
    # Offsets for Transmission
    trans_offsets = {'Automatic': 0.35, 'Manual': 0.0}
    trans_effect = np.array([trans_offsets[tr] for tr in transmission_sample])
    
    # Thermodynamic & Inertial Fuel Consumption Formula
    base_consumption = (
        1.15 * engine_size + 
        0.0024 * vehicle_weight + 
        0.0075 * horsepower - 
        0.18 * acceleration - 
        0.10 * (model_year - 2012) + 
        fuel_effect + 
        trans_effect
    )
    
    # Random Gaussian noise for real-world variation
    noise = np.random.normal(0, 0.45, n_samples)
    fuel_consumption = np.round(base_consumption + noise, 2)
    fuel_consumption = np.clip(fuel_consumption, 3.2, 17.8)
    
    # Build DataFrame
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
    
    return df

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "vehicle_fuel_dataset.csv")
    
    df = generate_vehicle_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Vehicle fuel dataset successfully generated at: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset Summary Head:")
    print(df.head())
    print("\nTarget Summary Statistics:")
    print(df['Fuel Consumption'].describe())

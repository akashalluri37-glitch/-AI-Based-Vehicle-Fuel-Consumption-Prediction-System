# 🚗 AI-Based Vehicle Fuel Consumption Prediction System

An end-to-end Machine Learning educational and automotive engineering project that predicts vehicle fuel consumption (**L/100 km** & **MPG**) based on engine specifications, vehicle mass, transmission, fuel type, and performance telemetry.

---

## 🎯 Problem Statement

In mechanical and automotive engineering, vehicle fuel consumption is a critical metric for vehicle design, fuel economy optimization, and emissions regulation.

This project delivers a complete Machine Learning workflow and interactive Streamlit web dashboard that answers:
> *"Given a vehicle's mechanical characteristics (Engine Size, Cylinders, Horsepower, Weight, Acceleration, Fuel Type, Transmission), how much fuel will it consume?"*

---

## 📊 Dataset Features & Specifications

| Feature | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| **Engine Size** | Numerical | 1.0 — 6.0 L | Engine displacement volume |
| **Cylinders** | Categorical/Discrete | 3, 4, 6, 8, 12 | Number of engine cylinders |
| **Horsepower** | Numerical | 70 — 500 HP | Engine power rating |
| **Vehicle Weight** | Numerical | 900 — 2800 kg | Vehicle curb mass |
| **Acceleration** | Numerical | 3.5 — 15.0 s | 0 to 100 km/h acceleration time |
| **Model Year** | Numerical | 2012 — 2024 | Vehicle production year |
| **Fuel Type** | Categorical | Petrol, Diesel, Hybrid, Plug-in Hybrid | Type of fuel system |
| **Transmission** | Categorical | Automatic, Manual | Gearbox type |
| **Fuel Consumption** | **Target** | 3.2 — 17.8 L/100 km | Fuel consumed per 100 km |

---

## 🤖 Regression Models Performance Benchmark

| Model Engine | MAE (L/100 km) | RMSE (L/100 km) | $R^2$ Score | Performance Rank |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.5001 | 0.6504 | 0.9622 | Linear Baseline |
| **Decision Tree Regressor** | 0.5987 | 0.7735 | 0.9466 | Non-linear Tree |
| **Random Forest Regressor** | **0.4288** | **0.5659** | **0.9714** | **Best Performance (97.14%)** |

---

## 📁 Project Architecture

```
vehicle_fuel_consumption_prediction/
├── data/
│   ├── generate_data.py               # Standalone dataset generator
│   └── vehicle_fuel_dataset.csv       # Historical vehicle dataset (1,200 rows)
├── models/
│   ├── saved_models/                  # Serialized joblib models & encoders
│   │   ├── regressor_linear_regression.joblib
│   │   ├── regressor_decision_tree.joblib
│   │   ├── regressor_random_forest.joblib
│   │   ├── preprocessor.joblib
│   │   └── evaluation_metrics.json
│   ├── model.joblib                   # Bundled best model & preprocessor package
│   └── train_models.py                # Wrapper executing src/train_model.py
├── notebooks/
│   └── vehicle_fuel_consumption.ipynb # Educational step-by-step Jupyter Notebook
├── src/                               # 🌟 Core Modular Pipeline (Educational & Production)
│   ├── __init__.py                    # Package initialization marker
│   ├── download_data.py               # Automated data acquisition & physics fallback generator
│   ├── data_loader.py                 # Feature engineering, scaling & train/test splits
│   └── train_model.py                 # Model training, benchmarking & artifact serialization
├── utils/
│   └── helpers.py                     # Rating categorizer, cost calculator & UI styles
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Dependency list
└── README.md                          # Comprehensive documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download / Generate Dataset
Acquire or regenerate the automotive dataset:
```bash
python src/download_data.py
```

### 3. Verify Data Loading & Preprocessing
Inspect features, One-Hot Encoding, StandardScaler, and train/test splits:
```bash
python src/data_loader.py
```

### 4. Train Models & Benchmark
Train Linear Regression, Decision Tree, and Random Forest, outputting MAE, RMSE, and $R^2$:
```bash
python src/train_model.py
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/vehicle_fuel_consumption.ipynb
```

### 6. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 💡 Student & Curriculum Learning Outcomes

1. **Regression Analysis**: Understanding continuous target variable modeling, MAE, RMSE, and $R^2$ variance metrics.
2. **Feature Preprocessing & Encoding**: Applying `StandardScaler` to numerical inputs and `OneHotEncoder` to categorical inputs (`Fuel Type`, `Transmission`).
3. **Automotive Physics Connection**: Relating vehicle weight inertia, engine displacement pumping losses, and aerodynamics to fuel consumption.
4. **Feature Importance**: Analyzing how vehicle weight and engine displacement dominate fuel consumption predictions.
5. **Interactive Web Deployment**: Building an end-to-end interactive dashboard with fuel rating badges, MPG conversions, and financial cost calculators.

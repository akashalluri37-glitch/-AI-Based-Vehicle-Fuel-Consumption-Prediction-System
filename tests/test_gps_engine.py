# ==============================================================================
# Script: test_gps_engine.py
# Purpose: Unit and regression tests for GPS Telemetry & Fuel Engine
# ==============================================================================

import os
import io
import unittest
import pandas as pd
import numpy as np

from utils.gps_engine import (
    haversine_distance,
    interpolate_route_waypoints,
    PRESET_ROUTES,
    generate_route_telemetry,
    predict_gps_route_fuel,
    generate_eco_route_alternatives,
    parse_custom_gps_csv
)

class TestGPSEngine(unittest.TestCase):

    def test_haversine_distance(self):
        # NYC (40.7128, -74.0060) to Boston (42.3601, -71.0589) ~ 306 km great circle
        dist = haversine_distance(40.7128, -74.0060, 42.3601, -71.0589)
        self.assertAlmostEqual(dist, 306.0, delta=10.0)

    def test_preset_routes_telemetry_generation(self):
        for route_key, info in PRESET_ROUTES.items():
            df_tel = generate_route_telemetry(route_key, n_waypoints=50)
            self.assertEqual(len(df_tel), 50)
            self.assertIn('Latitude', df_tel.columns)
            self.assertIn('Longitude', df_tel.columns)
            self.assertIn('Elevation_m', df_tel.columns)
            self.assertIn('Slope_pct', df_tel.columns)
            self.assertIn('Speed_kmh', df_tel.columns)
            self.assertGreater(df_tel['Distance_km'].iloc[-1], 0)

    def test_predict_gps_route_fuel_physics(self):
        # Test uphill mountain route consumes more fuel than flat route
        route_key = "🏔️ San Francisco ➔ Lake Tahoe (Mountain Climb)"
        df_tel = generate_route_telemetry(route_key, n_waypoints=60)
        
        vehicle_specs = {
            'Vehicle Weight': 1600,
            'Fuel Type': 'Petrol',
            'Engine Size': 2.0,
            'Horsepower': 170,
            'Transmission': 'Automatic'
        }
        base_ml_rate = 7.5  # L/100km
        
        df_res, summary = predict_gps_route_fuel(df_tel, vehicle_specs, base_ml_rate)
        
        # In a mountain climb route, total avg L/100km should exceed flat baseline due to elevation + grade
        self.assertGreater(summary['trip_avg_l100km'], base_ml_rate)
        self.assertGreater(summary['total_fuel_liters'], 0.0)
        self.assertGreater(summary['total_cost_usd'], 0.0)
        self.assertGreater(summary['total_co2_kg'], 0.0)
        self.assertTrue(10 <= summary['efficiency_score'] <= 100)

    def test_eco_route_alternatives(self):
        route_key = "🚗 New York City ➔ Boston (I-95 Corridor)"
        eco_routes = generate_eco_route_alternatives(route_key, {}, base_l100km=8.0)
        
        self.assertIn("🌿 Eco-Friendly Green Route", eco_routes)
        self.assertIn("⚡ Fast Highway Route", eco_routes)
        self.assertIn("🏙️ Shortest Urban Route", eco_routes)
        
        eco_fuel = eco_routes["🌿 Eco-Friendly Green Route"]["fuel_liters"]
        hwy_fuel = eco_routes["⚡ Fast Highway Route"]["fuel_liters"]
        
        # Eco route should consume less fuel than high speed highway route
        self.assertLess(eco_fuel, hwy_fuel)

    def test_custom_csv_parser(self):
        csv_data = """lat,lon,elevation,speed
37.7749,-122.4194,15,45
37.8044,-122.2711,45,95
37.8715,-122.2730,120,105
"""
        csv_file = io.StringIO(csv_data)
        df_clean = parse_custom_gps_csv(csv_file)
        self.assertEqual(len(df_clean), 3)
        self.assertIn('Segment_Distance_km', df_clean.columns)
        self.assertIn('Slope_pct', df_clean.columns)

if __name__ == '__main__':
    unittest.main()

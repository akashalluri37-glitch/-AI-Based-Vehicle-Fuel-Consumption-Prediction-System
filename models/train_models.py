# ==============================================================================
# Script: models/train_models.py
# Purpose: Model training entry point (Backward-compatible wrapper for src.train_model)
# ==============================================================================

import os
import sys

# Ensure root directory is in sys.path so src can be imported cleanly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.train_model import train_and_evaluate_models

def train_and_evaluate():
    """
    Executes the modular training pipeline from src/train_model.py.
    Maintains compatibility with existing scripts and entry points.
    """
    return train_and_evaluate_models()

if __name__ == "__main__":
    train_and_evaluate()

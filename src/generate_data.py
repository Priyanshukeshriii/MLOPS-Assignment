"""Generate the artificial dataset used in the assignment demonstration."""
from pathlib import Path
from src.data import generate_synthetic_data
from src.train import ROOT_DIR

if __name__ == "__main__":
    target = ROOT_DIR / "data" / "raw" / "housing.csv"
    data = generate_synthetic_data(target, n_rows=1200, random_state=42)
    print(f"Generated {len(data)} synthetic records at {target}")

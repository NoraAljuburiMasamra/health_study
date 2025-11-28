
import pandas as pd

def load_data(path):
    """Läser in CSV-data och returnerar en DataFrame."""
    return pd.read_csv(path) 

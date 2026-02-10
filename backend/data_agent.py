import pandas as pd
import os

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "nhanes_labeled.csv")

    data = pd.read_csv(data_path)
    return data

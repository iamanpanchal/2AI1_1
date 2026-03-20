import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- SECTION 1: LOAD DATA ---
# This part belongs to Member 1 (Lead)
df = pd.read_csv('insurance_data_linear.csv')

print("Dataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print(df.head())


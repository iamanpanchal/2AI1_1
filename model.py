import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- SECTION 1: LOAD DATA --
df = pd.read_csv('insurance_data_linear.csv')

print("Dataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print(df.head())

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("insurance_data_linear.csv")

# Select numerical features to scale
numerical_features = ["age", "bmi"]

# Apply scaling
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Check result
print("Scaled values of Age and BMI:")
print(data[["age", "bmi"]].head())
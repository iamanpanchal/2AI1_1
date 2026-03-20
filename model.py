import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- SECTION 1: LOAD DATA ---
df = pd.read_csv('insurance_data_linear.csv')

print("Dataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print(df.head())




# Member 3: Converting words to numbers (Encoding)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print("Categorical variables encoded")
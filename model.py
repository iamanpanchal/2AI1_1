import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

# Load dataset
df = pd.read_csv('insurance_data_linear.csv')  # apni file ka naam

# Ensure numeric
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')

# Plot
plt.figure(figsize=(10,6))
sns.histplot(df['charges'], kde=True, color='blue')

plt.title('Distribution of Insurance Charges')
plt.savefig('distribution.png')

print("EDA Visualizations created")
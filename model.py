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

# Member 5: Splitting the data
from sklearn.model_selection import train_test_split

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")

# Member 7: Evaluation

from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

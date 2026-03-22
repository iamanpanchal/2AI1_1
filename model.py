import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# MEMBER 1: LOAD DATA (Team Lead)
# ==========================================
df = pd.read_csv('insurance_data_linear.csv')
print("Dataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print(df.head())

# ==========================================
# MEMBER 2: EDA (Visualizations)
# ==========================================
plt.figure(figsize=(10,6))
sns.histplot(df['charges'], kde=True, color='blue')
plt.title('Distribution of Insurance Charges')
plt.savefig('distribution.png') 
# plt.show() # Uncomment if working in Colab
print("EDA Visualizations created")

# ==========================================
# MEMBER 3: ENCODING (Categorical to Numeric)
# ==========================================
# Convert sex, smoker, and region into numbers
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print("Categorical variables encoded")

# ==========================================
# MEMBER 4: SCALING (Standardization)
# ==========================================
scaler = StandardScaler()
num_cols = ['age', 'bmi', 'children']
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Features scaled using StandardScaler")

# ==========================================
# MEMBER 5: SPLITTING (Train/Test)
# ==========================================
X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")

# ==========================================
# MEMBER 6: TRAINING (Linear Regression)
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete")

# ==========================================
# MEMBER 7: EVALUATION (Metrics)
# ==========================================
y_pred = model.predict(X_test)

print("\n--- FINAL RESULTS ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
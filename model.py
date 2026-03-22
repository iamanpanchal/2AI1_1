import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# MEMBER 1: Data Loading (Team Lead)
# ==========================================
df = pd.read_csv('insurance_data_linear.csv')
print("--- Step 1: Data Loaded ---")
print(df.head())

# ==========================================
# MEMBER 2: Exploratory Data Analysis (EDA)
# ==========================================
# Visualize the distribution of the target variable 'charges'
plt.figure(figsize=(8, 5))
sns.histplot(df['charges'], kde=True, color='skyblue')
plt.title('Distribution of Medical Insurance Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# ==========================================
# MEMBER 3: Categorical Encoding
# ==========================================
# Convert text columns (sex, smoker, region) into numerical values
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print("\n--- Step 3: Encoding Complete ---")
print(df.head())

# ==========================================
# MEMBER 4: Feature Scaling
# ==========================================
# Scale numerical columns so they have a similar range
scaler = StandardScaler()
num_cols = ['age', 'bmi', 'children']
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\n--- Step 4: Feature Scaling Complete ---")

# ==========================================
# MEMBER 5: Data Splitting
# ==========================================
# Split the data into features (X) and target (y)
X = df.drop('charges', axis=1)
y = df['charges']

# 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n--- Step 5: Data Split (Train: {len(X_train)}, Test: {len(X_test)}) ---")

# ==========================================
# MEMBER 6: Model Training
# ==========================================
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n--- Step 6: Model Training Complete ---")

# ==========================================
# MEMBER 7: Evaluation & Results
# ==========================================
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- FINAL MODEL EVALUATION ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Optional: Visualization of Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Insurance Charges')
plt.show()
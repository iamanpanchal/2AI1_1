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
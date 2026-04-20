from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load and train the model once when the server starts
df = pd.read_csv('insurance_data_linear.csv') [cite: 4]
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
X = df.drop('charges', axis=1) [cite: 22]
y = df['charges']

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return "<h1>Insurance Prediction API is Running!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    # This is where you would get data from a form and return model.predict()
    return "Prediction Logic Goes Here"

if __name__ == "__main__":
    app.run(debug=True)
import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and train the model
df = pd.read_csv('insurance_data_linear.csv')
# Preprocessing
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Simple logic for the demo prediction
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0
    
    # We use a dummy row to match the 9 features the model expects
    prediction = model.predict([[age, bmi, children, 0, 0, smoker, 0, 0, 0]])[0]
    
    return render_template('index.html', prediction_text=f'Estimated Cost: ${prediction:,.2f}')

if __name__ == "__main__":
    # CRITICAL: This is what fixes the "No open ports detected" error
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- LOAD AND PREPARE MODEL ---
# We use a global variable for the model and the columns it expects
model = None
model_columns = None

def train_model():
    global model, model_columns
    # Skip bad lines to avoid merge conflict errors
    df = pd.read_csv('insurance_data_linear.csv', on_bad_lines='skip')
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    model = LinearRegression()
    model.fit(X, y)
    model_columns = list(X.columns)
    print("Model trained and columns saved!")

train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from user form
        input_dict = {
            'age': float(request.form['age']),
            'bmi': float(request.form['bmi']),
            'children': float(request.form['children']),
            'smoker_yes': 1 if request.form['smoker'] == 'yes' else 0
        }
        
        # 2. Create a DataFrame with all expected columns initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # 3. Fill in the values we have
        for col in input_dict:
            if col in input_df.columns:
                input_df[col] = input_dict[col]
        
        # 4. Predict
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', prediction_text=f'Estimated Insurance Cost: ${prediction:,.2f}')
    
    except Exception as e:
        # This will show the actual error on your webpage so you can debug
        return render_template('index.html', prediction_text=f"System Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

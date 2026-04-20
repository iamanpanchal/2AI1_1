import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- LOAD AND PREPARE MODEL ---
try:
    # Use 'on_bad_lines' to skip any conflict markers automatically
    df = pd.read_csv('insurance_data_linear.csv', on_bad_lines='skip')
    
    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    model = LinearRegression()
    model.fit(X, y)
    print("Model trained successfully!")
except Exception as e:
    print(f"Error during training: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from user
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        smoker_yes = 1 if request.form['smoker'] == 'yes' else 0
        
        # We create a simple prediction based on the expected columns
        # [age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]
        prediction = model.predict([[age, bmi, children, 0, smoker_yes, 0, 0, 0]])[0]
        
        return render_template('index.html', prediction_text=f'Estimated Cost: ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Calculation Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

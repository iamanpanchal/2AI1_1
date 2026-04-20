import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load Clean Data
df = pd.read_csv('insurance_data_linear.csv')
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
    try:
        data = [float(request.form['age']), float(request.form['bmi']), float(request.form['children'])]
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        # Placeholder for other encoded columns
        final_features = data + [0, smoker, 0, 0, 0] 
        prediction = model.predict([final_features])[0]
        return render_template('index.html', prediction_text=f'Cost: ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

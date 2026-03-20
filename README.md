# Medical Cost Prediction using Linear Regression
**Team Number:** 1  
**Project Objective:** Predict individual medical costs billed by health insurance.

## 👥 Team Members
* **Member 1 (Lead):** BOHARA MUFADDAL SHAKIR - Repo Setup & Coordination [DATASET LOADER]
* **Member 2:** AMAN PANCHAL - Exploratory Data Analysis (EDA)
* **Member 3:** ADYASA MAHARANA - Categorical Encoding
* **Member 4:** BANDAARU CHARANTEJA - Feature Scaling
* **Member 5:** BHAPKAR SHRUTI ANMOL - Data Splitting
* **Member 6:** C B HARSHAVARDHAN - Model Training
* **Member 7:** ANVESHA GUPTA - Model Evaluation & Review

---

## 📝 Problem Statement
The goal of this project is to use a Linear Regression model to predict the `charges` (medical insurance costs) based on several factors such as age, BMI, number of children, and smoking status.

## 📊 Dataset Description
The dataset used is the **Medical Cost Personal Dataset**. It contains 1,338 rows of data with the following features:
* **Age, Sex, BMI, Children, Smoker, Region** (Input features)
* **Charges** (Target variable we are predicting)

## ⚙️ Data Preprocessing Steps
1. **Loading:** Data was imported from a CSV file.
2. **Encoding:** Categorical variables (Sex, Smoker, Region) were converted to numerical values using One-Hot Encoding.
3. **Scaling:** Numerical features like Age and BMI were standardized using `StandardScaler`.
4. **Splitting:** The data was split into 80% Training and 20% Testing sets.

## 🤖 Model & Results
* **Model used:** Scikit-Learn Linear Regression.
* **Evaluation Metrics:** * Mean Squared Error (MSE): [Insert result here, e.g., 33,596,915]
  * R-squared ($R^2$) Score: [Insert result here, e.g., 0.78]

## 🤝 GitHub Collaboration Summary
Our team used a structured Git workflow:
* Each member worked on a dedicated **Feature Branch**.
* We utilized **Pull Requests** for every code addition.
* Peer reviews were conducted before merging into the `main` branch.

## 💡 Conclusion

The Linear Regression model successfully predicts medical costs with an accuracy of [X]%. Smoking status and BMI were found to be the most significant factors in determining insurance charges.
=======
The Linear Regression model successfully predicts medical costs with an accuracy of [X]%. Smoking status and BMI were found to be the most significant factors in determining insurance charges.

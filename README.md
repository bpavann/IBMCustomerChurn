# IBM Customer Churn Prediction

## Project Overview

This project is a Machine Learning web application that predicts whether a customer will churn (leave the service) or not. It is built using the IBM Telco Customer Churn dataset and deployed using Flask.

The main goal is to help businesses identify customers who are likely to leave so that retention strategies can be applied in advance.

---

## Problem Statement

Customer churn is a major problem in subscription-based businesses. Losing customers directly impacts revenue.

This project aims to:
- Analyze customer behavior data
- Build a machine learning model to predict churn
- Deploy the model as a web application for real-time prediction

---

## Dataset Information

The dataset contains customer information such as:

- Customer demographics (gender, senior citizen, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (internet service, phone service, streaming services)
- Billing details (monthly charges, total charges)

Target variable:
- Churn (Yes or No)

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- HTML, CSS (for frontend)
- Pickle (for model serialization)

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
   - Logistic Regression
   - Random Forest
   - Other classification models
6. Model Evaluation
7. Model Selection
8. Deployment using Flask

---

## How It Works

1. User enters customer details in the web form
2. Data is sent to Flask backend
3. Input is converted into a DataFrame
4. Preprocessing is applied using saved transformer
5. Trained model predicts churn
6. Result is displayed on the web page

---

## Model Output

The model predicts:
- 1 → Customer will churn
- 0 → Customer will not churn

In the application, this is converted into:
- Customer will CHURN
- Customer will NOT CHURN

---
## Conclusion

This project demonstrates how machine learning can be used to solve real business problems like customer churn. It combines data preprocessing, model building, and deployment into a complete end-to-end system.




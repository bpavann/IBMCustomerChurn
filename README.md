# IBM Customer Churn Prediction

## Project Overview

End-to-end Machine Learning web application that predicts whether a telecom customer will churn based on customer, service, account, and billing information.

The project covers data preprocessing, model training and evaluation, real-time prediction, containerization, and cloud deployment.

---

## Problem Statement

Predict whether a customer will leave the telecom service based on available customer and service information.

---

## Dataset

**IBM Telco Customer Churn Dataset**

The dataset includes:

* Customer demographics
* Account information
* Services subscribed
* Billing information

**Target:** `Churn` — Yes / No

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Flask
* HTML / CSS
* Docker
* Gunicorn
* Render

---

## Machine Learning Workflow

1. Data ingestion
2. Data cleaning and preprocessing
3. Exploratory Data Analysis
4. Feature transformation
5. Numerical and categorical preprocessing using `ColumnTransformer`
6. Model training
7. Model evaluation and comparison
8. Hyperparameter tuning
9. Best model selection
10. Model serialization
11. Real-time prediction through Flask

### Models Evaluated

* Logistic Regression
* KNN
* SVM
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* CatBoost
* AdaBoost

### Model Performance

The selected Logistic Regression model achieved:

* **Test Accuracy: 82.11%**
* **Train-Test Accuracy Gap: ~1.76%**

---

## Application Flow

```text
User Input
    ↓
Flask Web Application
    ↓
DataFrame
    ↓
Saved Preprocessor
    ↓
Trained ML Model
    ↓
Churn Prediction
    ↓
Web Result
```

The application returns:

* **Customer will CHURN**
* **Customer will NOT CHURN**

---

## Deployment
The application was containerized using **Docker** and served using **Gunicorn**.

### Live Application

**[IBM Customer Churn Prediction](https://ibmcustomerchurn.onrender.com)**

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model development to real-time Flask prediction, Docker containerization, cloud deployment on Render, and CI/CD preparation with Jenkins.

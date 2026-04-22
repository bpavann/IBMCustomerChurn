import os
import pickle
import pandas as pd
from flask import Flask, request, render_template
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictionPipeline

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            input_data = {
                "customerID": request.form.get('customerID'),
                "gender": request.form.get('gender'),
                "SeniorCitizen": int(request.form.get('SeniorCitizen') or 0),
                "Partner": request.form.get('Partner'),
                "Dependents": request.form.get('Dependents'),
                "tenure": int(request.form.get('tenure') or 0),
                "PhoneService": request.form.get('PhoneService'),
                "MultipleLines": request.form.get('MultipleLines'),
                "InternetService": request.form.get('InternetService'),
                "OnlineSecurity": request.form.get('OnlineSecurity'),
                "OnlineBackup": request.form.get('OnlineBackup'),
                "DeviceProtection": request.form.get('DeviceProtection'),
                "TechSupport": request.form.get('TechSupport'),
                "StreamingTV": request.form.get('StreamingTV'),
                "StreamingMovies": request.form.get('StreamingMovies'),
                "Contract": request.form.get('Contract'),
                "PaperlessBilling": request.form.get('PaperlessBilling'),
                "PaymentMethod": request.form.get('PaymentMethod'),
                "MonthlyCharges": float(request.form.get('MonthlyCharges') or 0.0),
                "TotalCharges": float(request.form.get('TotalCharges') or 0.0)
            }
            final_new_data=pd.DataFrame([input_data])
            logging.info(f"Final new data: {final_new_data}")
            print(final_new_data)

            prediction=PredictionPipeline()
            result=prediction.predict(final_new_data)
            logging.info(f"Prediction results: {result}")
            print(result)
            if result[0] == 1:
                output = "⚠️ Customer will CHURN"
            else:
                output = "✅ Customer will NOT CHURN"
            return render_template('home.html', result=output)
        
    except Exception as e:
        raise CustomException(e)

if __name__=="__main__":
    app.run(debug=True)
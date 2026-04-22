import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            logging.info("Loading preprocessor and model for prediction")
            preprocessor=load_object(file_path="artifacts/preprocessor.pkl")
            model=load_object(file_path="artifacts/model.pkl")
            logging.info("Transforming features and making prediction")
            data_transform=preprocessor.transform(features)
            pred=model.predict(data_transform)
            logging.info(f"Prediction completed: {pred}")
            return pred
        
        except Exception as e:
            raise CustomException(e)
         
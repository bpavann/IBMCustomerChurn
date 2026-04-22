import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def train_pipeline(self):
        try:
            logging.info("Starting the training pipeline")
            data_ingestion=DataIngestion()
            train_data, test_data=data_ingestion.init_data_ingestion()

            logging.info("Starting data transformation")
            data_transformation=DataTransformation()
            train_arr, test_arr, _= data_transformation.inti_data_transformation(train_data, test_data)

            logging.info("Starting model training")
            model_trainer=ModelTrainer()
            model_trainer.init_model_trainer(train_arr, test_arr)
            print(model_trainer.init_model_trainer(train_arr, test_arr))
            return
        except Exception as e:
            raise CustomException(e)
        
if __name__=="__main__":
    train=TrainPipeline()
    train.train_pipeline()
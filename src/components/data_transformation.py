import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from src.exception import CustomException
    from src.logger import logging
    from src.utils import save_object
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.exception import CustomException
    from src.logger import logging
    from src.utils import save_object

@dataclass
class DataTranformerConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTranformerConfig()

    def data_transformation_object(self, X):
        try:
            cat_feat = X.select_dtypes(include='object').columns
            num_feat = X.select_dtypes(exclude='object').columns

            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            logging.info(f"Categorical columns: {cat_feat}")
            logging.info(f"Numerical columns: {num_feat}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_feat),
                    ('cat_pipeline',cat_pipeline,cat_feat)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e)

    
    def inti_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info(f"Train and test data read successfully from {train_path} and {test_path}")

            for df in [train_df, test_df]:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = df['TotalCharges'].fillna(0)
                df.drop(['customerID'], axis=1, inplace=True)
                df['Churn']=df['Churn'].map({'Yes':1, 'No':0})
                logging.info("Data preprocessing completed successfully")
            
            # Define categorical and numerical columns
            target='Churn'
            X_train=train_df.drop(target, axis=1)
            y_train=train_df[target]
            X_test=test_df.drop(target,axis=1)
            y_test=test_df[target]
            logging.info("Separated features and target variable for training and testing data")

            preprocessor_obj=self.data_transformation_object(X_train)
            logging.info("Preprocessor object created successfully")

            X_train_arr= preprocessor_obj.fit_transform(X_train)
            X_test_arr= preprocessor_obj.transform(X_test)
            logging.info("Data transformation completed successfully")

            train_arr=np.c_[X_train_arr, np.array(y_train)]
            test_arr=np.c_[X_test_arr, np.array(y_test)]
            logging.info("Combined features and target variable into arrays for training and testing data")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info(f"Preprocessor object saved successfully at {self.data_transformation_config.preprocessor_obj_file_path}")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
try:
    from src.exception import CustomException
    from src.logger import logging
    from src.components.data_transformation import DataTransformation
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.exception import CustomException
    from src.logger import logging
    from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join("artifacts", "raw_data.csv")
    train_data_path=os.path.join("artifacts", "train_data.csv")
    test_data_path=os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Entered the data ingestion ")
        try:
            df=pd.read_csv("/Users/pavankumarb/Documents/My Learning/IBMCustomerChurn/notebook/dataset/Telco_Customer_Churn.csv")
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train test split initated")
            train_data,test_data=train_test_split(df,test_size=0.2, random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.init_data_ingestion()

    data_trans=DataTransformation()
    train_arr, test_arr,_=data_trans.inti_data_transformation(train_data, test_data)
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from src.componenets.data_transformation import DataTransformation,DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_excel("data\online_retail_II.xlsx")
            logging.info('read the excel file as Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('spliting train and test data')

            df = df.sort_values('InvoiceDate').reset_index(drop=True)

            test_size = int(len(df) * 0.2)
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

        


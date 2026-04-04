import os
import sys
import dill 
import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging

numerical_columns = [
                    'Price',
                    'month',
                    'hour',
                    'day_of_week',
                    'is_weekend',
                    'unique_items',
                    'order_value_bucket',
                    'quantity_bucket',
                    'avg_cust_price',
                    'avg_order_value',
                    'spending_tier',
                    'item_per_order_ratio',
                    'is_high_value',
                    'order_in_same_hour']
 
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            dill.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        logging.info('entered the load_object function')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        logging.info('model successfully loaded')

    except Exception as e:
        raise CustomException(e,sys)
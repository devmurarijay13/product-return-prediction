import os
import sys
import dill 
import numpy as np
import pandas as pd

from src.exception import CustomException

numerical_columns = ['Quantity',
                    'Price',
                    'order_value',
                    'month',
                    'hour',
                    'day_of_week',
                    'is_weekend',
                    'cust_return_rate',
                    'stock_return_rate',
                    'unique_items',
                    'cust_recency',
                    'return_recency',
                    'order_value_bucket',
                    'quantity_bucket',
                    'avg_cust_price',
                    'avg_order_value',
                    'spending_tier',
                    'return_ratio',
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
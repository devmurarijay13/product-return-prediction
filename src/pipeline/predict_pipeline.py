import pandas as pd
import numpy as np
import sys
import os
from src.utils import load_object
from src.exception import CustomException

class FastApiDataTransformation:
    def __init__(self):
        # Load the fitted preprocessor and the feature mapping dictionary
        self.preprocessor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
        self.feature_store = load_object(os.path.join('artifacts', 'feature_store.pkl'))

    def transform_predict_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Accepts a pandas DataFrame (converted from FastAPI JSON payload).
        Transforms it using strictly historical reference data.
        """
        try:
            fs = self.feature_store  # Shortcut to feature store
            
            # 1. Static Features
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Quantity'] = df['Quantity'].abs()
            df['Price'] = df['Price'].abs()
            df['order_value'] = df['Price'] * df['Quantity']

            df['month'] = df['InvoiceDate'].dt.month
            df['hour'] = df['InvoiceDate'].dt.hour
            df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # 2. Map Business Impact Features using historical data
            df['Country'] = df['Country'].apply(lambda x: x if x in fs['top_countries'] else 'Other')
            
            # Use fillna() extensively so new customers/products don't crash the system
            df['cust_return_rate'] = df['Customer ID'].map(fs['cust_rr_map']).fillna(fs['global_mean'])
            df['stock_return_rate'] = df['StockCode'].map(fs['stock_rr_map']).fillna(fs['global_mean'])
            
            # If standard payload lacks 'unique_items', default to 1 for API inference
            df['unique_items'] = 1 
            
            df['cust_recency'] = df['Customer ID'].map(fs['cust_recency_map']).fillna(999)
            df['return_recency'] = df['Customer ID'].map(fs['return_recency_map']).fillna(365)
            
            # Map Bins safely using pd.cut
            df['order_value_bucket'] = pd.cut(df['order_value'], bins=fs['ov_bins'], labels=False, include_lowest=True).fillna(0)
            df['quantity_bucket'] = pd.cut(df['Quantity'], bins=fs['q_bins'], labels=False, include_lowest=True).fillna(0)
            
            df['avg_cust_price'] = df['Customer ID'].map(fs['avg_price_map']).fillna(df['Price'])
            df['avg_order_value'] = df['Customer ID'].map(fs['cust_aov_map']).fillna(df['order_value'])
            
            labels = [1, 2, 3, 4, 5]
            df['spending_tier'] = pd.cut(df['avg_order_value'], bins=fs['tier_bins'], labels=labels, include_lowest=True).fillna(3).astype(int)
            
            df['return_ratio'] = df['Customer ID'].map(fs['final_ratio_map']).fillna(fs['global_mean'])
            df['item_per_order_ratio'] = df['unique_items'] / df['avg_order_value']
            df['is_high_value'] = (df['Price'] > fs['price_90th']).astype(int)
            
            # API batches usually process one event at a time. Defaulting to 1.
            df['order_in_same_hour'] = 1 

            # 3. Drop unneeded cols and Handle NaNs/Infs
            cols_to_drop = ['Invoice', 'StockCode', 'Description', 'InvoiceDate', 'Customer ID']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            # 4. Transform using preprocessor (NEVER fit_transform here)
            transformed_data = self.preprocessor.transform(df)
            
            return transformed_data

        except Exception as e:
            raise CustomException(e, sys)
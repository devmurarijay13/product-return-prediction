import sys
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as prep_pipe
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, numerical_columns

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    feature_store_file_path = os.path.join('artifacts', 'feature_store.pkl') # Added Feature Store Path

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_static_features(self, df):
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # ==========================================
            # THE "TIME TRAVEL" LABELING FIX
            # ==========================================
            # 1. Isolate the refund receipts (Invoices starting with 'C' or negative Quantity)
            returns_df = df[(df['Invoice'].str.startswith('C', na=False)) | (df['Quantity'] < 0)]
            
            # 2. Extract unique Customer-Product combinations that resulted in a return
            returned_pairs = returns_df[['Customer ID', 'StockCode']].drop_duplicates()
            returned_pairs['is_return'] = 1 # This is our actual target label
            
            # 3. Filter the main dataset to ONLY contain actual, original purchases
            df = df[~df['Invoice'].str.startswith('C', na=False)]
            df = df[df['Quantity'] > 0]
            df = df[df['Price'] > 0]
            
            # 4. Merge the 'is_return' label back onto the original purchase rows
            # If the Customer bought this StockCode and later returned it, it gets a 1. Otherwise, 0.
            df = df.merge(returned_pairs, on=['Customer ID', 'StockCode'], how='left')
            df['is_return'] = df['is_return'].fillna(0).astype(int)
            # ==========================================

            # Calculate order_value (safe to do now since we dropped negative quantities)
            df['order_value'] = df['Price'] * df['Quantity']

            # Extract temporal features
            df['month'] = df['InvoiceDate'].dt.month
            df['hour'] = df['InvoiceDate'].dt.hour
            df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Drop rows without Customer IDs
            df.dropna(subset=['Customer ID'], inplace=True)
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_business_impact_features(self, train_df, test_df):
        # Initialize Feature Store dictionary to save for API inference
        feature_store = {}

        global_mean = train_df['is_return'].mean()
        feature_store['global_mean'] = global_mean

        cust_stats = train_df.groupby('Customer ID')['is_return'].agg(['sum', 'count'])
        train_df = train_df.merge(cust_stats, on='Customer ID', how='left')
        train_df['cust_return_rate'] = (train_df['sum'] - train_df['is_return']) / (train_df['count'] - 1)
        train_df['cust_return_rate'].fillna(global_mean, inplace=True) 
        
        # Save historical customer return rates
        cust_rr_map = train_df.groupby('Customer ID')['is_return'].mean().to_dict()
        feature_store['cust_rr_map'] = cust_rr_map
        test_df['cust_return_rate'] = test_df['Customer ID'].map(cust_rr_map).fillna(global_mean)

        train_df.drop(columns=['sum', 'count'], inplace=True)

        train_mean = train_df['is_return'].mean()
        stock_rr = train_df.groupby('StockCode')['is_return'].mean()
        feature_store['stock_rr_map'] = stock_rr.to_dict()
        
        train_df['stock_return_rate'] = train_df['StockCode'].map(stock_rr)
        test_df['stock_return_rate'] = test_df['StockCode'].map(stock_rr).fillna(train_mean)

        unique_product = train_df.groupby('Invoice')['StockCode'].nunique().to_dict()
        train_df['unique_items'] = train_df['Invoice'].map(unique_product)
        test_df['unique_items'] = test_df['Invoice'].map(unique_product).fillna(1)

        curr_date = train_df['InvoiceDate'].max()
        last_purchase = train_df.groupby('Customer ID')['InvoiceDate'].max()
        cust_recency_map = (curr_date - last_purchase).dt.days.to_dict()
        feature_store['cust_recency_map'] = cust_recency_map

        train_df['cust_recency'] = train_df['Customer ID'].map(cust_recency_map).fillna(999)
        test_df['cust_recency'] = test_df['Customer ID'].map(cust_recency_map).fillna(999)

        top_countries = train_df['Country'].value_counts().head(10).index.tolist()
        feature_store['top_countries'] = top_countries
        train_df['Country'] = train_df['Country'].apply(lambda x: x if x in top_countries else 'Other')
        test_df['Country'] = test_df['Country'].apply(lambda x: x if x in top_countries else 'Other')

        rr_map = (curr_date - train_df[train_df['is_return']==1].groupby('Customer ID')['InvoiceDate'].max()).dt.days.to_dict()
        feature_store['return_recency_map'] = rr_map
        train_df['return_recency'] = train_df['Customer ID'].map(rr_map).fillna(365)
        test_df['return_recency'] = test_df['Customer ID'].map(rr_map).fillna(365)

        # FIX: Extract bins from train, apply via pd.cut to test
        train_df['order_value_bucket'], ov_bins = pd.qcut(train_df['order_value'], q=5, labels=False, duplicates='drop', retbins=True)
        feature_store['ov_bins'] = ov_bins
        test_df['order_value_bucket'] = pd.cut(test_df['order_value'], bins=ov_bins, labels=False, include_lowest=True)

        train_df['quantity_bucket'], q_bins = pd.qcut(train_df['Quantity'], q=4, labels=False, duplicates='drop', retbins=True)
        feature_store['q_bins'] = q_bins
        test_df['quantity_bucket'] = pd.cut(test_df['Quantity'], bins=q_bins, labels=False, include_lowest=True)

        avg_price = train_df.groupby('Customer ID')['Price'].mean().to_dict()
        feature_store['avg_price_map'] = avg_price
        train_df['avg_cust_price'] = train_df['Customer ID'].map(avg_price)
        test_df['avg_cust_price'] = test_df['Customer ID'].map(avg_price).fillna(train_df['Price'].mean())

        cust_aov = train_df.groupby('Customer ID')['order_value'].mean()
        feature_store['cust_aov_map'] = cust_aov.to_dict()
        train_df['avg_order_value'] = train_df['Customer ID'].map(cust_aov)
        test_df['avg_order_value'] = test_df['Customer ID'].map(cust_aov).fillna(train_df['order_value'].mean()) 

        labels = [1, 2, 3, 4, 5]
        train_df['spending_tier'], tier_bins = pd.qcut(train_df['avg_order_value'], q=5, labels=labels, retbins=True, duplicates='drop')
        feature_store['tier_bins'] = tier_bins
        test_df['spending_tier'] = pd.cut(test_df['avg_order_value'], bins=tier_bins, labels=labels, include_lowest=True).astype(int)
        train_df['spending_tier'] = train_df['spending_tier'].astype(int)

        total_invoices = train_df.groupby('Customer ID')['Invoice'].nunique()
        return_invoices_count = train_df[train_df['is_return'] == 1].groupby('Customer ID')['Invoice'].nunique()
        train_df['total_inv_count'] = train_df['Customer ID'].map(total_invoices)
        train_df['ret_inv_count'] = train_df['Customer ID'].map(return_invoices_count).fillna(0)
        
        train_df['return_ratio'] = (train_df['ret_inv_count'] - train_df['is_return']) / (train_df['total_inv_count'] - 1)
        global_ratio = return_invoices_count.sum() / total_invoices.sum()
        train_df['return_ratio'] = train_df['return_ratio'].replace([np.inf, -np.inf], np.nan).fillna(global_ratio)
        
        final_ratio_map = (return_invoices_count / total_invoices).fillna(0).to_dict()
        feature_store['final_ratio_map'] = final_ratio_map
        test_df['return_ratio'] = test_df['Customer ID'].map(final_ratio_map).fillna(global_ratio)
        
        train_df.drop(columns=['total_inv_count', 'ret_inv_count'], inplace=True)

        train_df['item_per_order_ratio'] = train_df['unique_items'] / train_df['avg_order_value']
        test_df['item_per_order_ratio'] = test_df['unique_items'] / test_df['avg_order_value']

        price_90th = train_df['Price'].quantile(0.9)
        feature_store['price_90th'] = price_90th
        train_df['is_high_value'] = (train_df['Price'] > price_90th).astype(int)
        test_df['is_high_value'] = (test_df['Price'] > price_90th).astype(int)

        train_df['order_in_same_hour'] = train_df.groupby(['Customer ID', 'InvoiceDate'])['Invoice'].transform('count')
        test_df['order_in_same_hour'] = test_df.groupby(['Customer ID', 'InvoiceDate'])['Invoice'].transform('count')

        cols_to_drop = ['Invoice', 'StockCode', 'Description', 'InvoiceDate', 'Customer ID']
        train_df.drop(columns=cols_to_drop, inplace=True)
        test_df.drop(columns=cols_to_drop, inplace=True)

        train_df, test_df = [d.replace([np.inf, -np.inf], np.nan).fillna(0) for d in [train_df, test_df]]

        return train_df, test_df, feature_store # Return the store

    def get_data_transformer_object(self):
        # ... [Unchanged: Keep your ColumnTransformer logic exactly as is] ...
        try:
            cat_col = ['Country']
            num_cols = numerical_columns

            cat_pipeline = prep_pipe(steps=[('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
            num_pipeline = prep_pipe(steps=[('scaler', RobustScaler())])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('categorical_pipeline', cat_pipeline, cat_col),
                    ('numerical_pipeline', num_pipeline, num_cols)
                ], remainder='drop'
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def resampling(self):
        # ... [Unchanged: Keep your resampler logic exactly as is] ...
        pipeline = Pipeline([
            ('adasyn', ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=3)),
            ('under_sampler', RandomUnderSampler(sampling_strategy='majority', random_state=42))
        ])
        return pipeline
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self.get_static_features(train_df)
            test_df = self.get_static_features(test_df)

            # Extract the feature store from the modified function
            train_df, test_df, feature_store = self.get_business_impact_features(train_df, test_df)

            target_col = 'is_return'
            X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
            X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

            preprocessor = self.get_data_transformer_object()
            resampler = self.resampling()

            X_train_trf = preprocessor.fit_transform(X_train)
            X_test_trf = preprocessor.transform(X_test)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_trf, y_train)

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_trf, np.array(y_test)]

            # Save the preprocessor
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)
            
            # Save the Feature Store! (Crucial for API)
            save_object(file_path=self.data_transformation_config.feature_store_file_path, obj=feature_store)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
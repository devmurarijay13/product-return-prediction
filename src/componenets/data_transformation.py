import sys
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as prep_pipe
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,numerical_columns

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_static_features(self,df):
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # 1. Identify Return Invoices (The Labels)
            # A 'C' invoice means the original purchase was returned.
            df['is_return'] = ((df['Invoice'].str.startswith('C', na=False)) | (df['Quantity'] < 0)).astype(int)

            # 2. Convert values to absolute to remove the "Negative Sign" cheat
            df['Quantity'] = df['Quantity'].abs()
            df['Price'] = df['Price'].abs()
            df['order_value'] = df['Price'] * df['Quantity']
            df['order_value'] = df['order_value'].abs()

            # 3. CRITICAL: Remove the actual return rows ('C' invoices) from the features
            # We want the model to learn from the original PURCHASE, not the RETURN row.
            df = df[df['Price'] > 0]
            df = df[df['Quantity'] > 0]

            df['month'] = df['InvoiceDate'].dt.month
            df['hour'] = df['InvoiceDate'].dt.hour
            df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            df.dropna(subset=['Customer ID'], inplace=True)

            return df

        except Exception as e:
            raise CustomException(e,sys)

    def get_business_impact_features(self,train_df,test_df):

       # 1. Calculate Global Mean (to fill for new customers)
        global_mean = train_df['is_return'].mean()

        # 2. Use a "Leave-One-Out" style calculation for Training
        # We subtract the current row's effect so the model can't cheat
        cust_stats = train_df.groupby('Customer ID')['is_return'].agg(['sum', 'count'])
        
        # Mapping back to train
        train_df = train_df.merge(cust_stats, on='Customer ID', how='left')
        
        # The trick: (Total Returns - Current Return) / (Total Orders - 1)
        train_df['cust_return_rate'] = (train_df['sum'] - train_df['is_return']) / (train_df['count'] - 1)
        train_df['cust_return_rate'].fillna(global_mean, inplace=True) # Handle customers with 1 order
        
        # 3. For Test set, use the raw mean from Train (this is okay because it's "Past Data")
        test_df['cust_return_rate'] = test_df['Customer ID'].map(train_df.groupby('Customer ID')['is_return'].mean()).fillna(global_mean)

        # Drop the helper sum/count columns before returning
        train_df.drop(columns=['sum', 'count'], inplace=True)

        train_mean = train_df['is_return'].mean()

        stock_rr = train_df.groupby('StockCode')['is_return'].mean()
        train_df['stock_return_rate'] = train_df['StockCode'].map(stock_rr)
        test_df['stock_return_rate'] = test_df['StockCode'].map(stock_rr)
        test_df['stock_return_rate'].fillna(train_mean,inplace=True)

        unique_product = train_df.groupby('Invoice')['StockCode'].nunique().to_dict()
        train_df['unique_items'] = train_df['Invoice'].map(unique_product)
        test_df['unique_items'] = test_df['Invoice'].map(unique_product)
        test_df['unique_items'].fillna(1,inplace=True)

        curr_date = train_df['InvoiceDate'].max()
        last_purchase = train_df.groupby('Customer ID')['InvoiceDate'].max()
        cust_recency_map = (curr_date - last_purchase).dt.days.to_dict()

        train_df['cust_recency'] = train_df['Customer ID'].map(cust_recency_map).fillna(999)
        test_df['cust_recency'] = test_df['Customer ID'].map(cust_recency_map).fillna(999)

        top_countries = train_df['Country'].value_counts().head(10).index.tolist()
        train_df['Country'] = train_df['Country'].apply(lambda x: x if x in top_countries else 'Other')
        test_df['Country'] = test_df['Country'].apply(lambda x: x if x in top_countries else 'Other')

        rr_map = (curr_date - train_df[train_df['is_return']==1].groupby('Customer ID')['InvoiceDate'].max()).dt.days.to_dict()
        # Days since last return
        train_df['return_recency'] = train_df['Customer ID'].map(rr_map).fillna(365)
        test_df['return_recency'] = test_df['Customer ID'].map(rr_map).fillna(365)

        train_df['order_value_bucket'] = pd.qcut(train_df['order_value'],q=5,labels=False)
        test_df['order_value_bucket'] = pd.qcut(test_df['order_value'],q=5,labels=False)

        train_df['quantity_bucket'] = pd.qcut(train_df['Quantity'],q=4,labels=False)
        test_df['quantity_bucket'] = pd.qcut(test_df['Quantity'],q=4,labels=False)

        avg_price = train_df.groupby('Customer ID')['Price'].mean().to_dict()
        train_df['avg_cust_price'] = train_df['Customer ID'].map(avg_price)
        test_df['avg_cust_price'] = test_df['Customer ID'].map(avg_price)
        test_df['avg_cust_price'].fillna(train_df['Price'].mean(),inplace=True)

        cust_aov = train_df.groupby('Customer ID')['order_value'].mean()
        train_df['avg_order_value'] = train_df['Customer ID'].map(cust_aov)
        test_df['avg_order_value'] = test_df['Customer ID'].map(cust_aov) 

        test_df['avg_order_value'].fillna(train_df['order_value'].mean(),inplace=True)

        labels = [1,2,3,4,5]
        
        train_df['spending_tier'], bins = pd.qcut(train_df['avg_order_value'],
                                         q=5,
                                         labels=labels,
                                         retbins=True,
                                         duplicates='drop')

        test_df['spending_tier'] = pd.cut(test_df['avg_order_value'],
                                 bins=bins,
                                 labels=labels,
                                 include_lowest=True).astype(int)
        train_df['spending_tier'] = train_df['spending_tier'].astype(int)

        invoices = train_df.groupby('Customer ID')['Invoice'].nunique()  
        return_invoices = train_df[train_df['is_return'] == 1].groupby('Customer ID')['Invoice'].nunique()
        cust_return_ratio = (return_invoices / invoices).fillna(0).to_dict()

        total_invoices = train_df.groupby('Customer ID')['Invoice'].nunique()
        
        # Calculate how many UNIQUE invoices were returns for each customer
        # (Where is_return == 1)
        return_invoices_count = train_df[train_df['is_return'] == 1].groupby('Customer ID')['Invoice'].nunique()
        # 2. Map these totals back to the training dataframe
        train_df['total_inv_count'] = train_df['Customer ID'].map(total_invoices)
        train_df['ret_inv_count'] = train_df['Customer ID'].map(return_invoices_count).fillna(0)
        # 3. APPLY THE FIX: Subtract the current invoice's contribution
        # If the current row's invoice is a return, we subtract 1 from the numerator AND denominator
        # Logic: (Total Return Invoices - Is Current Invoice a Return?) / (Total Invoices - 1)
        train_df['return_ratio'] = (train_df['ret_inv_count'] - train_df['is_return']) / (train_df['total_inv_count'] - 1)
        # Handle cases where a customer only has 1 invoice (division by zero)
        global_ratio = return_invoices_count.sum() / total_invoices.sum()
        train_df['return_ratio'] = train_df['return_ratio'].replace([np.inf, -np.inf], np.nan).fillna(global_ratio)
        # 4. FOR TEST SET: Use the final historical ratio from the Train set
        # This is safe because the test set is "Future" data
        final_ratio_map = (return_invoices_count / total_invoices).fillna(0).to_dict()
        test_df['return_ratio'] = test_df['Customer ID'].map(final_ratio_map).fillna(global_ratio)
        # 5. Cleanup helper columns
        train_df.drop(columns=['total_inv_count', 'ret_inv_count'], inplace=True)

        train_df['item_per_order_ratio'] = train_df['unique_items'] / train_df['avg_order_value']
        test_df['item_per_order_ratio'] = test_df['unique_items'] / test_df['avg_order_value']

        train_df['is_high_value'] = (train_df['Price']>train_df['Price'].quantile(0.9)).astype(int)
        test_df['is_high_value'] = (test_df['Price']>test_df['Price'].quantile(0.9)).astype(int)

        train_df['order_in_same_hour'] = train_df.groupby(['Customer ID','InvoiceDate'])['Invoice'].transform('count')
        test_df['order_in_same_hour'] = test_df.groupby(['Customer ID','InvoiceDate'])['Invoice'].transform('count')

        cols_to_drop = [
                'Invoice','StockCode','Description','InvoiceDate','Customer ID'
            ]

        train_df.drop(columns=cols_to_drop,inplace=True)
        test_df.drop(columns=cols_to_drop,inplace=True)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_df, test_df = [d.replace([np.inf, -np.inf], np.nan).fillna(0) for d in [train_df, test_df]]

        return train_df,test_df
    
    def get_data_transformer_object(self):
        try:
            cat_col = ['Country']
            num_cols = numerical_columns

            cat_pipeline = prep_pipe(
                steps=[
                    ('encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]
            )

            num_pipeline = prep_pipe(
                steps=[
                    ('scaler',RobustScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('categorical_pipeline',cat_pipeline,cat_col),
                    ('numerical_pipeline',num_pipeline,num_cols)
                ],
                remainder='drop'
            )
            logging.info('preprocessing pipeline completed')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def resampling(self):

        pipeline = Pipeline([
            ('adasyn',ADASYN(sampling_strategy='minority',random_state=42,n_neighbors=3)),
            ('under_sampler',RandomUnderSampler(sampling_strategy='majority',random_state=42))
        ])

        logging.info('resampling pipeline created')

        return pipeline
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read the Train and Test df')

            train_df = self.get_static_features(train_df)
            test_df = self.get_static_features(test_df)
            logging.info('static features created')

            train_df,test_df = self.get_business_impact_features(train_df,test_df)

            target_col = 'is_return'

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            preprocessor = self.get_data_transformer_object()
            resampler = self.resampling()
            logging.info('load the preprocessor and resampler')

            X_train_trf = preprocessor.fit_transform(X_train)
            X_test_trf = preprocessor.transform(X_test)
            logging.info('preprocessing applied')

            X_train_resampled,y_train_resampled = resampler.fit_resample(X_train_trf,y_train)
            logging.info('resampling applied')

            train_arr = np.c_[
                X_train_resampled,np.array(y_train_resampled)
            ]

            test_arr = np.c_[
                X_test_trf,np.array(y_test)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)


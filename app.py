import pandas as pd
import numpy as np
import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.utils import load_object
from src.exception import CustomException

# Input Data Schema (Pydantic)
class InvoiceData(BaseModel):
    Invoice: str
    StockCode: str
    Description: str
    Quantity: int
    InvoiceDate: str 
    Price: float
    # Pydantic doesn't allow spaces in variable names, so we use an alias
    Customer_ID: float = Field(alias="Customer ID")
    Country: str

class FastApiDataTransformation:
    def __init__(self):
        # Load the fitted preprocessor and the feature mapping dictionary
        self.preprocessor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
        self.feature_store = load_object(os.path.join('artifacts', 'feature_store.pkl'))

    def transform_predict_data(self, df: pd.DataFrame) -> np.ndarray:
        try:
            fs = self.feature_store  
            
            # 1. Static Features
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            df['Quantity'] = df['Quantity'].abs()

            df['Price'] = df['Price'].abs()

            df['order_value'] = df['Price'] * df['Quantity']

            df['month'] = df['InvoiceDate'].dt.month
            df['hour'] = df['InvoiceDate'].dt.hour
            df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # 2. Map Basic Business Impact Features
            df['Country'] = df['Country'].apply(lambda x: x if x in fs['top_countries'] else 'Other')

            df['cust_return_rate'] = df['Customer ID'].map(fs['cust_rr_map']).fillna(fs['global_mean'])

            df['stock_return_rate'] = df['StockCode'].map(fs['stock_rr_map']).fillna(fs['global_mean'])

            df['unique_items'] = 1 

            df['cust_recency'] = df['Customer ID'].map(fs['cust_recency_map']).fillna(999)

            df['return_recency'] = df['Customer ID'].map(fs['return_recency_map']).fillna(365)
            
            df['avg_cust_price'] = df['Customer ID'].map(fs['avg_price_map']).fillna(df['Price'])

            df['avg_order_value'] = df['Customer ID'].map(fs['cust_aov_map']).fillna(df['order_value'])

            safe_order_value = df['order_value'].clip(upper=fs['ov_bins'][-1])
            df['order_value_bucket'] = pd.cut(safe_order_value, bins=fs['ov_bins'], labels=False, include_lowest=True).fillna(4)

            # Clip quantity
            safe_quantity = df['Quantity'].clip(upper=fs['q_bins'][-1])
            df['quantity_bucket'] = pd.cut(safe_quantity, bins=fs['q_bins'], labels=False, include_lowest=True).fillna(3)
            
            safe_aov = df['avg_order_value'].clip(upper=fs['tier_bins'][-1])
            labels = [1, 2, 3, 4, 5]
            df['spending_tier'] = pd.cut(safe_aov, bins=fs['tier_bins'], labels=labels, include_lowest=True).fillna(5).astype(int)
            
            df['return_ratio'] = df['Customer ID'].map(fs['final_ratio_map']).fillna(fs['global_mean'])

            df['item_per_order_ratio'] = df['unique_items'] / df['avg_order_value']

            df['is_high_value'] = (df['Price'] > fs['price_90th']).astype(int)

            df['order_in_same_hour'] = 1 

            cols_to_drop = ['Invoice', 'StockCode', 'Description', 'InvoiceDate', 'Customer ID']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            # 6. Transform using preprocessor
            transformed_data = self.preprocessor.transform(df)
            
            return transformed_data

        except Exception as e:
            raise CustomException(e, sys)


app = FastAPI(title="Return Prediction API")

# Initialize transformer and load the ML model at startup
transformer = FastApiDataTransformation()

try:
    model = load_object(os.path.join('artifacts', 'model.pkl'))
except Exception as e:
    print("Warning: model.pkl not found. Please ensure your trained model is saved.")
    model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Return Prediction API"}

@app.post("/predict")
def predict_return(data: InvoiceData):
    try:
        # Convert Pydantic object to a dictionary (respecting the "Customer ID" alias)
        data_dict = data.model_dump(by_alias=True) if hasattr(data, 'model_dump') else data.dict(by_alias=True)
        
        df = pd.DataFrame([data_dict])
        
        ## Transforming the data
        transformed_features = transformer.transform_predict_data(df)
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")
            
        prob = model.predict_proba(transformed_features)[0][1]
        
        prediction = int(prob >= 0.50) 
        
        return {
            "prediction": prediction,
            "probability": round(float(prob), 4),
            "status": "Success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
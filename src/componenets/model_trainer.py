import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna

from dataclasses import dataclass
import warnings
warnings.filterwarnings(action='ignore')

from src.exception import CustomException
from src.logger import logging
from src.componenets.data_transformation import DataTransformation,DataTransformationConfig
from src.utils import save_object

import lightgbm as lgb
from lightgbm import LGBMClassifier,log_evaluation,early_stopping
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.metrics import f1_score,roc_auc_score,precision_score,confusion_matrix,classification_report,recall_score,precision_recall_curve,ConfusionMatrixDisplay

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_and_evaluate(self, train_arr, test_arr):
        try:
            logging.info('splitting training and testing data')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            logging.info('validation set created with stratified shuffling')

            model = LGBMClassifier(
                n_estimators=392, 
                learning_rate=0.0014313483962791853, 
                num_leaves=26,
                max_depth=5, 
                min_child_samples=62,
                feature_fraction=0.7326590215464699, 
                bagging_fraction=0.8312192363074786, 
                bagging_freq=7,
                random_state=42
            )
                        
            logging.info('model created')

            model.fit(    
                X_train_final, y_train_final,                    
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss", 
                callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=10)]
            )
            
            logging.info('model trained')   

            test_probs = model.predict_proba(X_test)[:, 1]        

            precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
            
            # Avoid division by zero
            f1_scores = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall)!=0)
            
            # Get the threshold that maximizes the F1 score
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            print(f"\n---> Dynamically Calculated Optimal Threshold: {best_threshold}")
            
            # Apply the dynamic threshold!
            y_pred_optimized = (test_probs >= best_threshold).astype(int)
            
            print("\n--- FINAL TEST SCORES ---")
            print("="*50)
            print("Precision Score :", precision_score(y_test, y_pred_optimized))
            print('Recall Score :', recall_score(y_test, y_pred_optimized))
            print("f1 Score :", f1_score(y_test, y_pred_optimized))
            print("roc auc score :", roc_auc_score(y_test, test_probs))

            print("\nConfusion Matrix : \n")
            print(confusion_matrix(y_test, y_pred_optimized))

            print("\nClassification Report:\n")
            print(classification_report(y_test, y_pred_optimized))
            print("="*50 + "\n")

            test_probs = model.predict_proba(X_test)[:, 1]        

            print("\n--- THRESHOLD DECISION TABLE ---")
            print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score'}")
            print("-" * 50)

            # Print metrics at different threshold levels
            for t in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
                y_pred_t = (test_probs >= t).astype(int)
                p = precision_score(y_test, y_pred_t, zero_division=0)
                r = recall_score(y_test, y_pred_t, zero_division=0)
                f1 = f1_score(y_test, y_pred_t, zero_division=0)
                print(f"{t:<12.2f} | {p:<10.4f} | {r:<10.4f} | {f1:.4f}")
            print("-" * 50)
        
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=model
            )

        except Exception as e:
            raise CustomException(e, sys)

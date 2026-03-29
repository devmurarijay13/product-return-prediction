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

    def train_and_evaluate(self,train_arr,test_arr):
        try:

            logging.info('spliting training and testing data')
            X_train, y_train, X_test, y_test = (
                    train_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, :-1],
                    test_arr[:, -1],
                )
            tscv = TimeSeriesSplit(n_splits=5)
        
            for train_index, val_index in tscv.split(X_train):
                X_train_final, X_val = X_train[train_index], X_train[val_index]
                y_train_final, y_val = y_train[train_index], y_train[val_index]

            logging.info('validation set has created from training data to avoide data leakage')

                # -------------------------------------------------------------------------------
                        
            model = LGBMClassifier(
                  n_estimators=392, learning_rate=0.0014313483962791853, 
                  num_leaves=26,max_depth= 5, min_child_samples= 62,
                  feature_fraction= 0.7326590215464699, bagging_fraction= 0.8312192363074786, 
                  bagging_freq=7, scale_pos_weight= 35.00166103513742
                )
                        
            logging.info('model created')

            model.fit(    
                X_train_final, y_train_final,                    
                eval_set=[(X_val, y_val)],
                eval_metric="f1",
                callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=0)]
                )
            
            logging.info('model trained')   


            test_probs = model.predict_proba(X_test)[:, 1]        

            precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            # best_threshold = thresholds[np.argmax(f1_scores)]

            # print(f"Optimal Threshold for F1: {best_threshold}")
            y_pred_optimized = (test_probs >= 0.6571502206705265).astype(int)
                
            print("\n--- FINAL TEST SCORES ---")
                
            logging.info('model evaluation summary')
            print("="*50 + "\n")
            print("Precision Score :",precision_score(y_test,y_pred_optimized))
            print('Recall Score :',recall_score(y_test,y_pred_optimized))
            print("f1 Score :",f1_score(y_test,y_pred_optimized))
            print("roc auc score :",roc_auc_score(y_test,test_probs))

            print("Confusion Matrix : \n")
            print(confusion_matrix(y_test,y_pred_optimized))

            print(classification_report(y_test,y_pred_optimized))
            print("="*50 + "\n")

        
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=model
                )

        except Exception as e:
            raise CustomException(e,sys)

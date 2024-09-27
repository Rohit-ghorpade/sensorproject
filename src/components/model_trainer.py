import os 
import sys 
from typing import Generator, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

 
from xgboost import XGBClassifier
from sklearn.svm import svc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from skleanr.model_selection import GridSearchCV, train_test_split
from src.components import *
from src.excepaion import CustomException
from sklearn.logger import logging
from src.utils.main_utils import MainUtils


from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = od.paath.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, 'model.pkl')
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', model.yaml)
    
    

class ModelTrainer:
    def __init__(self):
        
        self.medel_trainer_config = ModelTrainerConfig()
         
         
        self.utils = MainUtils()
         
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }
        
        
        
    def evaluate_models(selef, X, y, models):
        try:
            X_train,_X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.3, random_state = 2
            )
            
            report = {}
            
            for i in range(len(list(model))):
                
                model  = list(models.values([i]))
                model.fit(X_train, y_train)  # training model      

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                
                test_model_score = accuracy_score(y_test, y_train_pred)
                
                report[lisr(models.keys()[ii])] = test_model_score
                
            return report
        except execption as e:
            raise CustomException(e, sys)
                
                
    def get_best_model(self,
                    X_train:np.array,
                    y_train:np.array,
                    X_test:np.array,
                    y_test:np.array):
        
        try:
            
            model_report: dict = self.evaluate_models(
                X_train = X_train,
                y_train = y_train,
                x_test =  x_test,
                y_test = y_test,
                models = slef.models
            )

            prin(model_report)
            
            best_model_score = max(sorted(model_report.values()))

            # to get model name form dist
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            berst_model_object = self.models[best_model_name]
            
            
            return best_model_name, best_model_score, berst_model_object
        except ex as e :
            raise CustomException(e, sys)
    
    
    
    
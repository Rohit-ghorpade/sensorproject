import sys
from typing import Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45  # Ensure consistency in the accuracy threshold
    model_config_file_path = os.path.join('config', 'model.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }

    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            report = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)  # Train model
                y_test_pred = model.predict(X_test)  # Predict on test set
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, x_train: np.array, y_train: np.array) -> Tuple[str, object, float]:
        try:
            model_report: dict = self.evaluate_models(x_train, y_train, self.models)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)  # More concise way to get the best model name
            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object: object, best_model_name: str, X_train: np.array, y_train: np.array) -> object:
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            logging.info(f"Best params for {best_model_name}: {best_params}")

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array: np.array, test_array: np.array) -> str:
        try:
            logging.info("Splitting training and testing input and target features")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            model_report: dict = self.evaluate_models(X=x_train, y=y_train, models=self.models)
            best_model_name, best_model, _ = self.get_best_model(x_train, y_train)

            best_model = self.finetune_best_model(best_model_name=best_model_name, best_model_object=best_model, X_train=x_train, y_train=y_train)

            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
            logging.info(f"Best model name: {best_model_name}, Score: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"No best model found with an accuracy greater than the threshold {self.model_trainer_config.expected_accuracy}")

            logging.info("Best found model on both training and testing dataset")
            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)

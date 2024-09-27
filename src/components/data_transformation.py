import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from src.constant import *  # Ensure TARGET_COLUMN is properly imported or declared
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder)  # Ensure artifact_folder is defined in src.constant
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        """Loads data from the given file path."""
        try:
            if not os.path.exists(feature_store_file_path):
                raise FileNotFoundError(f"The file {feature_store_file_path} does not exist.")
            
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)  # Ensure TARGET_COLUMN is correct
            return data 
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """Returns a preprocessor pipeline for data transformation."""
        try:
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                    imputer_step,
                    scaler_step
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """Performs data transformation and returns transformed arrays and preprocessor path."""
        logging.info("Entered initiate_data_transformation method of data transformation class")
        
        try:
            # Load the data
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)

            # Separate features (X) and target (y)
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Get preprocessor pipeline
            preprocessor = self.get_data_transformer_object()

            # Transform the data
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Save preprocessor object to a file
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

            # Concatenate the transformed features with target values
            train_arr = np.c_[X_train_scaled, np.array(y_train)]  # Fixed: np.c_[] for column-wise concatenation
            test_arr = np.c_[X_test_scaled, np.array(y_test)]     # Fixed: np.c_[]

            # Return transformed arrays and preprocessor path
            return (train_arr, test_arr, preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys) from e

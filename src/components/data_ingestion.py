import sys
import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from src.exception import CustomException
from src.utils.main_utils import MainUtils

class DataIngestionConfig:
    artifact_folder: str = os.path.join("artifacts")  # Ensure artifact_folder is correctly defined

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name: str, db_name: str) -> pd.DataFrame:
        """Exports a MongoDB collection to a Pandas DataFrame."""
        try:
            # Log the connection attempt
            logging.info(f"Connecting to MongoDB at {MONGO_DB_URL}")
            mongo_client = MongoClient(MONGO_DB_URL)

            collection = mongo_client[db_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            # Log the number of records fetched
            logging.info(f"Fetched {len(df)} records from {db_name}.{collection_name}")

            # Drop the '_id' column if it exists
            if "_id" in df.columns:
                df.drop(columns=['_id'], inplace=True)

            # Replace 'na' string with NaN
            df.replace({"na": np.nan}, inplace=True)

            # Check if DataFrame is empty
            if df.empty:
                raise CustomException("The DataFrame is empty. No records were found in the collection.")

            return df
        except Exception as e:
            logging.error(f"Error occurred while exporting collection: {e}")
            raise CustomException(e, sys)

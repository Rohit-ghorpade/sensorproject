import os

AWS_S3_BUCKET_NAME = "wafer-fault"
MONGO_DATABASE_NAME = "pwskills"
MONGO_COLLECTION_NAME = "waferfault"

TARGET_COLUMN = "quality"

# Load from environment variables
MONGO_DB_URL = os.getenv("mongodb+srv://rohit:ROHIT00@cluster0.xsg1l.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"

ARTIFACTS_DIR: str = "artifacts" 

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_DIR: str = "https://drive.google.com/file/d/1ekCoHan7JdbxP6ikVsDWKwiboBj5KMlw/view?usp=sharing"


"""
Data Validation related constant end with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = "status.txt"

DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "test", "valid", "data.yaml"]


"""
Model Trainer related constant end with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov8l.pt"

MODEL_TRAINER_NO_EPOCHS: int = 50

MODEL_TRAINER_BATCH_SIZE: int = 16
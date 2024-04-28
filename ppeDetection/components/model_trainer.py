import os, sys
from ppeDetection.logger import logging
from ppeDetection.exception import AppException
from ppeDetection.entity.config_entity import ModelTrainerConfig
from ppeDetection.entity.artifacts_entity import ModelTrainerArtifact
from ultralytics import YOLO


class ModelTrainer:
    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered intiate_model_trainer method of ModelTrainer class")
        
        try:
            logging.info("Unzipping data")
            os.system("unzip data.zip")
            os.system("rm data.zip")
            
            os.makedirs("yolov8l_train", exist_ok=True)
            os.system(f"cd yolov8l_train && yolo task=detect mode=train \
                      model={self.model_trainer_config.weight_name} \
                      imgsz=640 \
                      batch={self.model_trainer_config.batch_size} \
                      epochs={self.model_trainer_config.no_epochs} \
                      data='C:/Users/cjx14/Personal_Projects/PPE-Detection-YOLOv8/data.yaml'\
                      name='yolov8l_results'")
            
            os.system("cp yolov8l_train/runs/detect/yolov8l_results/weights/best.pt yolov8l_train/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp yolov8l_train/runs/detect/yolov8l_results/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
            
            os.system(f"rm -rf yolov8l_train/{self.model_trainer_config.weight_name}")
            os.system("rm -rf yolov8l_train/runs") 
            os.system("rm -rf train")
            os.system("rm -rf test")
            os.system("rm -rf valid")
            os.system("rm -rf data.yaml")
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov8l_train/best.pt"
            )
            
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
        except Exception as e:
            raise AppException(e, sys)
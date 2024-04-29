# PPE-Detection-YOLOv8

## Overview
This project utilizes the YOLOv8 model to detect personal protective equipment (PPE) in images, aiming to enhance safety measures in environments such as construction sites. It features a comprehensive machine learning pipeline that includes environment setup, data processing, model training, and deployment for real-time inference.

## Project Setup
### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- Compatible cuda toolkit and cudnn installed on your machine.
- Anaconda or Miniconda installed on your machine.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Kanon14/PPE-Detection-YOLOv8.git
cd PPE-Detection-YOLOv8
```

2. Create and activate a Conda environment:
```bash
conda create -n ppe python=3.8 -y
conda activate ppe
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
### Training Data Source
The training data is sourced from the [Construction Site Safety Computer Vision Project](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) provided by the Roboflow.

## Workflow
The project workflow is designed to facilitate a seamless transition from development to deployment:
1. `constants`: Manage all fixed variables and paths used across the project.
2. `entity`: Define the data structures for handling inputs and outputs within the system.
3. `components`: Include all modular parts of the project such as data preprocessing, model training, and inference modules.
4. `pipelines`: Organize the sequence of operations from data ingestion to the final predictions.
5. `app.py`: This is the main executable script that ties all other components together and runs the whole pipeline.
6. `live-ppe-detect.py`: Application for live detection using a webcam.

## How to Run
### Training and Image Detection:
1. Execute the project:
```bash
python app.py
```
2. Then, access the application via your web browser:
```bash
open http://localhost:<port>
```
### For Live PPE Detection with Webcam:
1. To initiate live detection:
```bash
python live-ppe-detect.py
```
2. Access the live application:
```bash
open http://localhost:<port>
```
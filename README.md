# PPE-Detection-YOLOv8

# PROJECT IN-PROGRESS!!!

## Project Setup
### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- YOLOv8 dependencies
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

## Workflow
The project's workflow is designed to facilitate a smooth transition from development to deployment:

1. `constants`: Manage all fixed variables and paths used across the project.
2. `entity`: Define the data structures for handling inputs and outputs within the system.
3. `components`: Include all modular parts of the project such as data preprocessing, model training, and inference modules.
4. `pipelines`: Organize the sequence of operations from data ingestion to the final predictions.
5. `app.py`: This is the main executable script that ties all other components together and runs the whole pipeline.
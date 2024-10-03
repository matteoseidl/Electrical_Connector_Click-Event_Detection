import sys
import os
from os.path import dirname, abspath
from pathlib import Path
import importlib
import torch
from torch import nn

model_architectures_dir = "03_Click_Detection_Model/01_modelArchitectures"
architecture_file = "ClickDetectorCNN_v1.py"

class ClickDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model_import()

    def model_import(self):
        #module = importlib.import_module(architecture_file[:-3])
        current_file_path = os.path.abspath(__file__)
        current_file_parent_dir = dirname(current_file_path)
        print(f"current_file_parent_dir: {current_file_parent_dir}")
        project_dir = dirname(current_file_parent_dir)
        print(f"project_dir: {project_dir}")
        model_architectures_dir_path = os.path.join(project_dir, model_architectures_dir)
        if os.path.exists(model_architectures_dir_path):
            sys.path.append(model_architectures_dir_path)
            model_module = importlib.import_module(architecture_file[:-3])
            ClickDetectorCNN = getattr(model_module, 'ClickDetectorCNN') #access the ClickDetectorCNN class
            model = ClickDetectorCNN(input_channels=1, output_shape=1).to(self.device)
            return model
        else:
            print("Model architectures directory does not exist")
        return None
    
if __name__ == '__main__':
    clickdetector = ClickDetector()
    model = clickdetector.model_import()
    print(model)

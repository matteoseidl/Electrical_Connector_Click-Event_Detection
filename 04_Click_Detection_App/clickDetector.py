import sys
import os
from os.path import dirname, abspath
from pathlib import Path
import importlib
import torch
from torch import nn

model_architectures_dir = "03_Click_Detection_Model/01_modelArchitectures"
selected_model = "ClickDetectorCNN_v1"
model_weights_path = "03_Click_Detection_Model/02_savedWeights/ethernet_det_model_0.pt"   

class ClickDetector:
    def __init__(self):
        self.model_architectures_dir = model_architectures_dir
        self.model_weights_path = model_weights_path
        self.selected_model = selected_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()

    def load_model(self):
        current_file_path = os.path.abspath(__file__)
        current_file_parent_dir = dirname(current_file_path)
        print(f"current_file_parent_dir: {current_file_parent_dir}")
        project_dir = dirname(current_file_parent_dir)
        print(f"project_dir: {project_dir}")
        model_architectures_dir_path = os.path.join(project_dir, self.model_architectures_dir)
        model_weights = os.path.join(project_dir, self.model_weights_path)
        if os.path.exists(model_architectures_dir_path):
            sys.path.append(model_architectures_dir_path)
            model_module = importlib.import_module(self.selected_model)
            ClickDetectorCNN = getattr(model_module, 'ClickDetectorCNN') #access the ClickDetectorCNN class
            model = ClickDetectorCNN(input_channels=1, output_shape=1).to(self.device)
            if os.path.exists(model_weights):
                model.load_state_dict(torch.load(model_weights))
                return model
            else:
                print("Model weights file does not exist")
        else:
            print("Model architectures directory does not exist")
        return None
    
    def load_model_weights(self, model, model_weights_path):
        pass
    
    def normalize_spec_chunk(self, spec_chunk):
        dB_min = -100
        dB_max = 0
        normalized_spec_chunk = (spec_chunk - dB_min) / (dB_max - dB_min)
        return normalized_spec_chunk
    
    def convert_to_torch_tensor(self, spec_chunk):
        spec_chunk_tensor = torch.from_numpy(spec_chunk).type(torch.float32).unsqueeze(0).unsqueeze(0) #add batch and channel dimensions
        return spec_chunk_tensor
    
    def detection(self, model, spec_chunk):
        model.eval()
        binary_threshold = 0.5
        with torch.inference_mode():
            model_prediction = model(spec_chunk)
            model_prediction = torch.squeeze(model_prediction)
            print(model_prediction)
        
        binary_predictions = (model_prediction > binary_threshold).float()
        print(binary_predictions)
        
        return binary_predictions
        

if __name__ == '__main__':
    clickdetector = ClickDetector()
    model = clickdetector.model_import()
    print(model)

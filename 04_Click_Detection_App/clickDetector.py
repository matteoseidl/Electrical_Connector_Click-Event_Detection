import sys
import os
from os.path import dirname, abspath
from pathlib import Path
import importlib
import torch
from torch import nn

class ClickDetector:
    def __init__(self):
        
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu" # use cpu for real time detection
    
    def load_model(self, model_architectures_dir, selected_model, model_weights_path):
        current_file_path = os.path.abspath(__file__)
        current_file_parent_dir = dirname(current_file_path)
        project_dir = dirname(current_file_parent_dir)
        model_architectures_dir_path = os.path.join(project_dir, model_architectures_dir)
        model_weights = os.path.join(project_dir, model_weights_path)

        if os.path.exists(model_architectures_dir_path):
            sys.path.append(model_architectures_dir_path)
            model_module = importlib.import_module(selected_model)
            ClickDetectorCNN = getattr(model_module, 'ClickDetectorCNN') #access the ClickDetectorCNN class
            model = ClickDetectorCNN(input_channels=1, output_shape=1).to(self.device)
            if os.path.exists(model_weights):
                model.load_state_dict(torch.load(model_weights, map_location=self.device, weights_only=True)) # load model weights, map location of weights to device
                model.to(self.device)
                print("Model weights have been loaded")
                print(f"model: {model}")
                return model
            else:
                print("Model weights file does not exist")
        else:
            print("Model architectures directory does not exist")

        return None
    
    def normalize_spec_chunk(self, spec_chunk):
        # min and max dB values for normalization
        dB_min = -120
        dB_max = 0
        normalized_spec_chunk = (spec_chunk - dB_min) / (dB_max - dB_min)

        return normalized_spec_chunk
    
    def convert_to_torch_tensor(self, spec_chunk):
        spec_chunk_tensor = torch.from_numpy(spec_chunk).type(torch.float32).unsqueeze(0).unsqueeze(0) #add batch and channel dimensions

        return spec_chunk_tensor
    
    def detection(self, model, spec_chunk):
        model.eval()
        binary_threshold = 0.5 # threshold for binary classification
        with torch.inference_mode():
            model_prediction = model(spec_chunk)
            model_prediction = torch.squeeze(model_prediction)
            print(model_prediction)
        
        binary_predictions = (model_prediction > binary_threshold).float() # binary classification based on threshold
        print(binary_predictions)
        
        return binary_predictions

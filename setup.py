import os
import subprocess
import sys

## dictionaries to create
directories = [
    '01_Dataset',
    '01_Dataset/01_audioDatasets',
    '01_Dataset/01_audioDatasets/01_Ethernet',
    '01_Dataset/01_audioDatasets/02_Ethernet_Test',
    '01_Dataset/01_audioDatasets/03_HVA280',
    '01_Dataset/01_audioDatasets/04_HVA280_Test',
    '01_Dataset/01_audioDatasets/05_HVA630',
    '01_Dataset/01_audioDatasets/06_HVA630_Test',
    '01_Dataset/01_audioDatasets/07_Noise_Samples',

    '02_Data_Augmentation',
    '02_Data_Augmentation/01_augmentedDatasets',
    '02_Data_Augmentation/01_augmentedDatasets/01_Ethernet',
    '02_Data_Augmentation/01_augmentedDatasets/02_HVA280',
    '02_Data_Augmentation/01_augmentedDatasets/03_HVA630',
    '02_Data_Augmentation/01_augmentedDatasets/04_Noise_Samples',

    '03_Click_Detection_Model',
    '03_Click_Detection_Model/01_modelArchitectures',
    '03_Click_Detection_Model/02_savedWeights',
    '03_Click_Detection_Model/03_trainingResults',

    '04_Detection_Model_Test',
    '04_Detection_Model_Test/01_testResults',
]

## create the directories if they do not exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    else:
        print(f"{directory} already exists.")

print("Required directories created!")

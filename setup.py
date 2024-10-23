import os
import subprocess
import sys

# missing dictionaries to create
directories = [
    '01_Dataset/01_audioDatasets',
    '01_Dataset/01_audioDatasets/01_Ethernet',
    '01_Dataset/01_audioDatasets/02_Ethernet_Test',
    '01_Dataset/01_audioDatasets/03_HVA280',
    '01_Dataset/01_audioDatasets/04_HVA280_Test',
    '01_Dataset/01_audioDatasets/05_HVA630',
    '01_Dataset/01_audioDatasets/06_HVA630_Test',
    '01_Dataset/01_audioDatasets/07_Noise_Samples',

    '02_Data_Augmentation/01_augmentedDatasets',
    '02_Data_Augmentation/01_augmentedDatasets/01_Ethernet',
    '02_Data_Augmentation/01_augmentedDatasets/02_HVA280',
    '02_Data_Augmentation/01_augmentedDatasets/03_HVA630',
    '02_Data_Augmentation/01_augmentedDatasets/04_Noise_Samples',

    '03_Click_Detection_Model/02_savedWeights',
]

# create the directories if they do not exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("Required directories created!")

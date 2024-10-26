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
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    else:
        print(f"{directory} already exists.")

print("Required directories created!")

"""# check if requirements.txt exists
if os.path.exists('requirements.txt'):
    # install dependencies from requirements.txt
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
else:
    print("requirements.txt not found. Required dependency are not installed.")"""

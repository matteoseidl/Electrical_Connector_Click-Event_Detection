import os
import subprocess
import sys

# missing dictionaries to create
directories = [
    '01_Dataset/01_ethernet_without_additional_noise',
    '01_Dataset/01_audioDatasets/02_ethernet_with_additional_noise',

    '02_Data_Augmentation/01_generated_datasets',

    '03_Click_Detection_Model/02_savedWeights',
]

# create the directories if they do not exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("Required directories created!")

# check if requirements.txt exists
if os.path.exists('requirements.txt'):
    # install dependencies from requirements.txt
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
else:
    print("requirements.txt not found. Required dependency are not installed.")

# File Dataloader_intro.py
# About: used to become familiar with image data loaders for pytorch

# Importing required libraries
import os
from regression_dataset import Regression_dataset
from torch.utils.data import DataLoader


####### Data Selection 
# The training process starts with selecting a folder containing the data for training.

# Get the current working directory (cwd)
datasets_filepath = "/home/pathfinder/Datasets/"


dataset_list = [] # List of our datasets, empty for now
for filename in os.listdir(datasets_filepath):
    if os.path.isdir(os.path.join(datasets_filepath, filename)): # checks if filename is for a folder
        dataset_list.append(filename) # If the filename is a folder, its added to the list of datasets
print("Available Datasets: ") 
print(dataset_list)

selected_data = dataset_list[2] # Default is the first dataset
print("Selected data for training: " + selected_data )
selected_dataset_filepath = os.path.join(datasets_filepath, selected_data)

# Loading the selected folder of datapoints into a custom dataset, Regression_dataset, created for image regression
train_dataset = Regression_dataset(selected_dataset_filepath,transform=None, random_hflip=False)
print("Data points in the training dataset: " + str(train_dataset.__len__())) # Size of the dataset

# Consider using a regular pytorch dataloader here, (take advantage of the index and the batching)
# Consider creating a verfied dataset to act as the test data.

# Prototype:
#train_dataloader = DataLoader(train_dataset, batch_size=100)

# Mock training process, used to demonstrate the iteration of the dataset
def simple_train(dataset):
    i = 0 # an index for the file number

    # For loop iterating across the given datset, producing the image and the label
    for images, xy in enumerate(dataset):

        if i % 1000 == 999:
            print("i: " + str(i))
            print("File name: " + str(images))

        i = i+1
    print(i)

simple_train(train_dataset)

print(train_dataset[0])
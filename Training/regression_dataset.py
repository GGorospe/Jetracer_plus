# File: regression_dataset.py
# Author: George Gorospe, george.gorospe@nmaia.net
# About: This is a custom dataloader to support an image regression task

# About the data: 
#   - The target data for this dataloader is images of any shape.
#   - The images should be of a format taht cv2.imread accepts.
#   - The labels for each image should be in the file name with the format xxx_yyy_*.*

# Annotations, how data about each datapoint is captured
# Each datapoint has an annotation in the annotations dictionary
# annotations structure: 
# annotations = [{
#                    'image_path': image_path,
#                    'category_index': category_index,
#                    'x': x,
#                    'y': y
#                }]


# Importing required libraries
import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np


class Regression_dataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(Regression_dataset, self).__init__()
        self.directory = directory
        self.annotations = [] # A currently empty list for annotations of type dict
        #self.categories = categories
        self.create_directory()
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    # Get Item function: input --> index of desired item within dataset
    # From the list of annotations retreive indexed file and read
    # PIL.Image.fromarray(image)
    # Transform image: if the transform flag is positive, applytransform, set new x and y.
    def __getitem__(self, idx):
        ann = self.annotations[idx]  # Retrieves the annotation (filelocation, x, and y values)
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        # The x and y values from the file name are pixel values, this calculation changes those
        # values to a range of -1 to +1, this is done for training purposes, these values will be stored as tensors
        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom

        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
        return image, torch.Tensor([x, y])
    
    # Parsing function: takes the filename (full path) and splits it to obtain x and y labels
    # Expects a data file name in the format xxx_yyy_*.*
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        return int(x), int(y)
    
    def create_directory(self):
        print("Creating New Dataset Directory: ")
        if not os.path.exists(self.directory):
            print("Success")
            subprocess.call(['mkdir', '-p', self.directory])
        else:
            print("Dataset directory exists.")
        
# Refresh list function: clears the list of annotations before cycling through the datafiles writing new annotations        
    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y
            }]
        
    def save_entry(self, new_directory_name, image, x, y):
        category_dir = os.path.join(self.directory, new_directory_name)
        if not os.path.exists(category_dir):
            subprocess.call(['mkdir', '-p', category_dir])
            
        filename = '%d_%d_%s.jpg' % (x, y, str(uuid.uuid1()))
        
        image_path = os.path.join(category_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
    
    
# Update Directory function: clears the list of annotations before cycling through the datafiles writing new annotations
    def update_directory(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y
            }]
        
    def get_count(self, category):
        i = 0
        for a in self.annotations:
            if a['category'] == category:
                i += 1
        return i


class HeatmapGenerator():
    def __init__(self, shape, std):
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap
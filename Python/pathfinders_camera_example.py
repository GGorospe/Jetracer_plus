# pathfinders_camera_example.py
# Author: George Gorospe, george.gorospe@nmaia.net

# Learning how to use python to control the jetracer
# This script shows how to collect a single photograph using the jetcam libary

# Importing required libraries
import os # This library is used to ensure that the saved file is in the same location as this script
import time # We use this libary for the 'sleep' function that pauses execution 
from uuid import uuid1    # This library is used to generate unique file names
from jetcam.csi_camera import CSICamera  # We use this libary to control the csi camera on the jetracer
from jetcam.utils import bgr8_to_jpeg    # The numpy array of blue, green and red values needs to be converted to a jpeg to be saved.


# Employing Object Oriented Programming (OOP) intantiate the camera object from the CSICamera Library module
camera = CSICamera(width=224, height=224)

time.sleep(5)


# Collect an image and modify it so that it may be saved as a .jpg file
# Note: When using the CSICamera module, collected images are of type numpy array. 
#       To save the data as an image we need to convert to jpeg format.
image = camera.read()        # camera.read() uses the read() method of the camera object
image = bgr8_to_jpeg(image)  # Convert data to a jpeg


# Create a new unique filename then save the file in the new directory.
uuid = '%s' % (uuid1()) #uuid(1) creates a unique string each time it's called
directory = os.getcwd(); # Use the os library to get the current location of this script.
image_path = os.path.join(directory, 'sample_image_' + uuid + '.jpg') # Building the path & filename for the new labeled data point

# Saving the image to file
with open(image_path, 'wb') as f: # f is now the name of the new file, but it doesn't have any content yet
    f.write(image) # Here we write the image information to the new file

print('New File Created: \n')
print(image_path)

# Shutdown camera and release resources
camera.running = False
# pathfinders_vehicle_servo_control_CLA.py
# Author: George Gorospe, george.gorospe@nmaia.net

# Learning how to use python to control the jetracer
# This script features a commanndline argument parser
# Using the command line argument and the NvidiaRacecar library, we can change the steering servo position

# Import the argparse library to manage command line arguments
from jetracer.nvidia_racecar import NvidiaRacecar
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# add a desired argument or 'flag' for the parser to look for
# In this case '--value' is the flag, and information that comes after the flag will be interpreted as a float value
# Finally, this argument is considered requried, this means that if the '--value' and ### is not included, an error will occur
parser.add_argument('--value', type=float, required=True) # Parser for servo command

# The action of taking the supplied input and producing the value within the args object.
args = parser.parse_args()

# Instantiate the object for Object Oriented Programming (OOP)
car = NvidiaRacecar()

# Commanding the jetracer's steering using the value supplied from the parser.
car.steering = args.value
print("The commanded servo value is: ", args.value)

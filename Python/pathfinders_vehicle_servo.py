# pathfinders_vehicle_servo_control.py
# Author: George Gorospe, george.gorospe@nmaia.net

# Learning how to use python to control the jetracer
# This program will send a single command to the Jetracer's steering servo
# Before executing this code, ensure that the jetracer's propulsion and control system is on, 
# Also ensure that the mux is set to jetson control

# Importing required libraries
from jetracer.nvidia_racecar import NvidiaRacecar

# Instantiate the object for Object Oriented Programming (OOP)
car = NvidiaRacecar()

# using the steering parameter of the NvidiaRacecar() object
car.steering = 0.3
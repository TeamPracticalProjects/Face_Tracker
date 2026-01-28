# Bridge_Testing
#
# (c) 20206, Team Practical projects, Bob Glicksman, Jim Schrempp; all rights reserved.

from arduino.app_utils import *
import time

print("\nStarting Bridge_Testing\n")

x_value = 200
y_value = 300

def loop():
    # x_value = input("Enter the x coordinate in pixel units: ")
    # y_value = input("Enter the y coodinate in pixel units: ")

    global x_value
    global y_value

    x_value = x_value + 1
    y_value = y_value + 1
  
    coordinate_string = f'{x_value},{y_value}'
    #coordinate_string = "150,-230"
    print("The coordinate string is: " + coordinate_string)  # print the string created to the Python console
    Bridge.call("move_servos", coordinate_string)

    time.sleep(2)

App.run(user_loop=loop)
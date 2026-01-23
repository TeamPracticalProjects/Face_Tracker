# Bridge_Testing

Generate integer x and y coordinate data.  Assemble the coordinate data into a string in
the form of "x_value,y_value" and use the bridge to transfer this string to the Arduino side.

The Arduino side receives the string and uses the Monitor to print it out.  It also flashes the onboard LED.

A future enchancement is to parse the received string and control two servos of a pan/tilt mechanism.

## Bricks Used

**This example does not use any Bricks.** It shows direct Router Bridge communication between Python® and Arduino.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)

### Software

- Arduino App Lab


## How to Use the test code

1. Run the App

2. Observe the Arduino LED to see that the string came across the bridge.  Use the Python console and the Arduino console to view the
values that are sent over the brdige.

4. In a future enhancement, send pixel coodinates as x and y values and observe pan/tilt mechanism movement.

## How it Works

### Python:
Integer x and y values are generated from base values by incrementing each by one for each pass through the loop() 
function.  Convert the integers to strings and assemble these into a single string in the form of "x,y" using 
Python f-strings.  Use the Bridge to send this composite string to the Arduino side.  Then, delay for 2 seconds and repeat the loop()

### Arduino: 
Set up the Bridge and Monitor (Arduino console).  "move_servos" is the Arduino function that is called from the Python side
of the Bridge.  This function receives a String argument from the brdige.  It prints the received string to the Arduino console via the
Monitor.println() function and flashes the built-in Arduino LED twice.



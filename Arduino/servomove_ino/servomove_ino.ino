/************************************************************************************
  servomove.ino:  Arduino software to move pan/tilt servo system based upon serial
    input in the form of x,y, where x and y are signed integers.

    The values of x and y are pixel values in image space, relative to the center
    of the image; i.e. 0,0 is the center of the image.  Positive x,y values are
    to the right and up; negative values are to the left and down.

    The pan/tilt mechanism is calibrated to the camera/sensor used to generate the
    image.  The defined value HORIZ_PIXELS is the number of pixels from the left to the
    right edge of the image.  The defined value FOV is the horizontal field of view of
    the camera/sensor.  Alternativle, the vertical image size and the verticle FOV may be
    used, as the pixels are assumed to be square.  This scaling mechanism is used to compute
    the number of degrees to move the servo in order to point the center of the mechanism
    to the designated point on the image.

    Two modes of operation are provided using the defined value of ABSOLUTE_MODE.  When
    ABSOLUTE_MODE is defined, the x,y values provided are used to drive the sevos so that
    the mechanism points to that image location.  ABSOLUTE_MODE is useful when the image source
    is fixed and the pan/tilt mechanism moves to point at a designated point in the image space
    (i.e. to point a paintball gun).  When ABSOLUTE_MODE is commented out, the software runs in
    RELATIVE_MODE.  RELATIVE_MODE is used when the camera is mounted on the pan/tilt mechanism
    and the mechanism is used to move the camera so as to place the designed point in the image
    to the center of the image.

    When using this software, be sure to edit the source code for your particular application:

      - #define DEBUG.  Comment out this line to surpress serial printing of the software calculations.

      - #define ABSOLUTE_MODE.  Comment out for RELATIVE_MODE,

      - const int HORIZ_PIXELS = 1280;.  Set to the width of the camera/sensor image in pixels.

      - const float FOV = 50.0;.  Set to the camera/sensor horizontal field of view (degrees).

    version 1.0, by Bob Glicksman, 12/26/25
    (c) 2025, 2026; Bob Glicksman, Jim Schrempp, Team Practical Projects.  All rights reserved.

************************************************************************************/

#define DEBUG   // for testing

#define ABSOLUTE_MODE // comment out for REALTIVE_MODE

#include <Servo.h>  // the servo library
#include <math.h>   // for math functions like round()

#define TILT_SERVO_PIN 2  //The Tilt servo is attached to pin 2
#define PAN_SERVO_PIN 3   //The Pan servo is attached to pin 3
#define LED_PIN 13

#define INITIAL_PAN_SERVO 90  // the initial position of the pan servo
#define INITIAL_TILT_SERVO 90 // the initial position of the tilt servo

const int HORIZ_PIXELS = 1280;  // camera matrix is 1280 x 720 (720p; 1Mpix)
const float FOV = 50.0;         // the camera specified field of view (degrees)

Servo servoTilt, servoPan;      // define the pan and tilt servo objects
float degreesPerPixel;          // conversion factor of pixels into servo movement degrees

// set the initial positions of the servos
//    must calibrate the mechanism with the camera in ABSOLUTE mode; not totally necessary in RELATIVE mode.
int panServoPosition = INITIAL_PAN_SERVO;
int tiltServoPosition = INITIAL_TILT_SERVO;  

void setup() {
  
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600); // initialize the serial port

  servoTilt.attach(TILT_SERVO_PIN);
  servoPan.attach(PAN_SERVO_PIN);

  //Initially position both servos to initial positions
  servoTilt.write(tiltServoPosition);  
  servoPan.write(panServoPosition);

  // compute conversion factor for turning pixels (+, - from center) into degrees based on pixel matrix and FOV of camera
  degreesPerPixel = FOV / HORIZ_PIXELS;
  #ifdef DEBUG
    Serial.print("degrees per pixel = ");
    Serial.println(degreesPerPixel);
    Serial.println("*** INITIALIZED ****\n");
  #endif

  // flash the onboad LED to signal end of setup()
  
  for(int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  digitalWrite(LED_PIN, HIGH);

  // flush out the serial buffer befor entering loop()
  while (Serial.available() > 0) {
    Serial.read();
  }

} // end of setup()

void loop() {

  int newPanServo, newTiltServo;  // the new positions to move the servos to
  int newX, newY;

  // read data from the serial port in the form of x,y where x and y are image coordinates relative to the center of the image

  while (Serial.available() == 0) {};  // wait for input character
  newX = Serial.parseInt(); 
  Serial.read();  // flush out the comma delimiter
  newY = Serial.parseInt();

  #ifdef DEBUG
    Serial.print("new X value = ");
    Serial.print(newX);
    Serial.print(" new Y value = ");
    Serial.print(newY);
    Serial.println("\n");
  #endif

  // convert the new X and Y image pixel values to servo angles
  #ifdef ABSOLUTE_MODE
    float newXServo = INITIAL_PAN_SERVO + (degreesPerPixel * newX);
    float newYServo = INITIAL_TILT_SERVO - (degreesPerPixel * newY);


  #else   // this would be RELATIVE mode
    float newXServo = panServoPosition + (degreesPerPixel * newX);
    float newYServo = tiltServoPosition - (degreesPerPixel * newY);

  #endif

  // the new servo positions
  newPanServo = round(newXServo);
  newTiltServo = round(newYServo);

  // constrain the servo values to be withon 0 - 180 degrees
  newPanServo = constrain(newPanServo, 0, 180);
  newTiltServo = constrain(newTiltServo, 0, 180);

  servoPan.write(newPanServo);
  servoTilt.write(newTiltServo);

  // reset the current servo states
  panServoPosition = newPanServo;
  tiltServoPosition = newTiltServo;

  #ifdef DEBUG
    Serial.print("pan servo position = ");
    Serial.print(panServoPosition);
    Serial.print(" ; tilt sevo position = ");
    Serial.print(tiltServoPosition);
    Serial.println("\n");
  #endif  

  // flush out the newline from the serial buffer before returning for a new value
  while (Serial.available() > 0) {
    Serial.read();
  }

} // end of loop()


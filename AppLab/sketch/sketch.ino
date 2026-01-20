/************************************************************************************
  servomove.ino: Arduino software to move pan/tilt servo system based upon serial
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

            1.1  * pulled operative code out of main into movePanTilt()
                 * added #define CALIBRATE. When set, input values are directly passed to servos without manipulation
                 * constants for PAN and TILT max/min so servos are not driven beyond the mechanism limits
                 * TILT_SERVO_REVERSED when 1 then servo moves up with smaller numbers and down with larger ones
                 * made FOV into two constants, horizontal and vertical for cameras with non-square views


    (c) 2025, 2026; Bob Glicksman, Jim Schrempp, Team Practical Projects.  All rights reserved.

************************************************************************************/

//#define DEBUG   // for testing
//#define CALIBRATE // if defined, then serial input is NOT scaled and passed directly to the servos

//#define ABSOLUTE_MODE // comment out for REALTIVE_MODE

#include <Servo.h>  // the servo library
#include <math.h>   // for math functions like round()

#define TILT_SERVO_PIN 2  //The Tilt servo is attached to pin 2
#define PAN_SERVO_PIN 3   //The Pan servo is attached to pin 3
#define LED_PIN 13

// These values constrain the servos so they do not move outside of the limits of the mechanism
// Set to 0 and 180 if the servos can freely move to the limits of their travel
#define PAN_SERVO_MINIMUM 0
#define PAN_SERVO_MAXIMUM 180
#define TILT_SERVO_MINIMUM 0
#define TILT_SERVO_MAXIMUM 140

#define TILT_SERVO_REVERSED 1   // set to 1 if tilt servo values decrease when the servo moves up

const int HORIZ_PIXELS = 320;  // camera matrix is 1280 x 720 (720p; 1Mpix), but the coordinates are based upon 320 x 180
const int VERT_PIXELS = 180;
const float HORIZ_FOV = 50; //100.0;         // the camera specified field of view (degrees)
const float VERT_FOV = 50; //70.0;         // the camera specified field of view (degrees)

Servo servoTilt, servoPan;      // define the pan and tilt servo objects
float degreesPerPixel;          // conversion factor of pixels into servo movement degrees

// set the initial positions of the servos
//    must calibrate the mechanism with the camera in ABSOLUTE mode; not totally necessary in RELATIVE mode.
int panServoMidPosition = PAN_SERVO_MINIMUM + (PAN_SERVO_MAXIMUM - PAN_SERVO_MINIMUM)/2;
int tiltServoMidPosition = TILT_SERVO_MINIMUM + (TILT_SERVO_MAXIMUM - TILT_SERVO_MINIMUM)/2;
int panServoPosition = panServoMidPosition; 
int tiltServoPosition = tiltServoMidPosition; 

// ********************************************
// move the pan/tilt mechanism constrained to the limits of
// the servos and the mechanism
void movePanTilt(int newX, int newY){
  int newPanServo, newTiltServo;  // the new positions to move the servos to

  // convert the new X and Y image pixel values to servo angles
  #ifdef ABSOLUTE_MODE
    if (TILT_SERVO_REVERSED) {
      newY = -newY;
    }
    float newXServo = map(newX, -HORIZ_PIXELS/2 , HORIZ_PIXELS/2, panServoMidPosition - HORIZ_FOV/2, panServoMidPosition + HORIZ_FOV/2 );
    float newYServo = map(newY, -VERT_PIXELS/2, VERT_PIXELS/2, tiltServoMidPosition - VERT_FOV/2, tiltServoMidPosition + VERT_FOV/2) ;  

  #else   // this would be RELATIVE mode
    float newXServo = panServoPosition + (degreesPerPixel * newX);
    float newYServo = tiltServoPosition - (degreesPerPixel * newY);

  #endif

  // the new servo positions
  newPanServo = newXServo;
  newTiltServo = newYServo;

  // constrain the servo values to be withon 0 - 180 degrees
  newPanServo = constrain(newPanServo, PAN_SERVO_MINIMUM, PAN_SERVO_MAXIMUM);
  newTiltServo = constrain(newTiltServo, TILT_SERVO_MINIMUM, TILT_SERVO_MAXIMUM);

  servoPan.write(newPanServo);
  servoTilt.write(newTiltServo);

  // reset the current servo states
  panServoPosition = newPanServo;
  tiltServoPosition = newTiltServo;

  #ifdef DEBUG
    Serial.print("pan servo position = ");
    Serial.print(panServoPosition);
    Serial.print(" ; tilt servo position = ");
    Serial.print(tiltServoPosition);
    Serial.println("\n");
  #endif  


}   // end of movePanTilt()


// ********************************************
void setup() {
  
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200); // initialize the serial port
  delay (2000); // allow time for serial port to initialize

  servoTilt.attach(TILT_SERVO_PIN);
  servoPan.attach(PAN_SERVO_PIN);

  //Initially position both servos to initial positions
  movePanTilt(0,0);
//  servoTilt.write(tiltServoPosition);  
//  servoPan.write(panServoPosition);

  // compute conversion factor for turning pixels (+, - from center) into degrees based on pixel matrix and FOV of camera
  degreesPerPixel = HORIZ_FOV / HORIZ_PIXELS;
  #ifdef DEBUG 
    Serial.println("\n\n\n");
    Serial.println("*** INITIALIZED ****\n");
    Serial.print("degrees per pixel = ");
    Serial.println(degreesPerPixel);
  #endif

  #ifdef DEBUG
    Serial.print("pan servo position = ");
    Serial.print(panServoPosition);
    Serial.print(" ; tilt servo position = ");
    Serial.print(tiltServoPosition);
    Serial.println("\n");
  #endif  

  Serial.println("Platform now centered in field of view.");
  Serial.print("Input x,y from ");
  Serial.print(-HORIZ_PIXELS/2);
  Serial.print(",");
  Serial.print(-VERT_PIXELS/2); 
  Serial.print(" to ");
  Serial.print(HORIZ_PIXELS/2);
  Serial.print(",");
  Serial.print(VERT_PIXELS/2); 
  Serial.print(" will map to a FOV of ");
  Serial.print(HORIZ_FOV);
  Serial.print(",");
  Serial.print(VERT_FOV);
  Serial.print(" degrees.");
  Serial.println("");

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


// ********************************************
void loop() {

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

  #ifdef CALIBRATE
    Serial.println("CALLIBRATE MODE");
    servoPan.write(newX);
    servoTilt.write(newY);
  #else
    // normal operation
    movePanTilt(newX,newY);
  #endif

  // flush out the newline from the serial buffer before returning for a new value
  while (Serial.available() > 0) {
    Serial.read();
  }

} // end of loop()


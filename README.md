# Face Tracker.

  This project controls a servo based pan/tilt mechanism to trace a face in view of a camera.  The project uses Python and OpenCV to acquire
  a face and track its position in realtime.  The Python code places a bounding rectangle on the detected face and sends the x,y
  image coordinates of the center of this bounding rectangle out over Serial I/O in the form of x,y where positive x is right, negative
  x is left, positive y is up and negative y is down, all relative to the center of the image.

  The pan/tilt mechanism is controlled by an Arduino Uno.  The Arduino software has two modes of operation.  In ABSOLUTE mode, the mechanism
  points to the x,y position received on the Serial I/O port.  In RELATIVE mode, the mechansim moves x,y reltaive to its current poition
  so as to keep the indicated image coordinates in the center of the camera field of view.  Absolute mode is useful where the camera is
  immobile and the mechanism points to the indicated position on the image.  Relative mode is useful when the camera is mounted on the
  mechanism and the mechanism tries to always keep the camera centered on the indicated point of the image (center of the face).

    ... THIS REPOSITORY IS CURRENTLY A WORK IN PROCESS ...
  

  

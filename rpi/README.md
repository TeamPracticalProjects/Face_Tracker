# Face Tracker - Raspberry Pi

Real-time face tracking system using OpenCV and servo control for Raspberry Pi. Tracks faces detected by the camera and controls pan/tilt servos to follow the largest detected face.

## Features

- **Real-time face detection** using OpenCV Haar Cascade classifiers
- **Multi-face support** - detects frontal and profile faces
- **Servo control** - pan/tilt servos follow the largest detected face
- **Web streaming** - view the camera feed with face detection overlays from any browser on your network
- **GPIO control** using pigpio for hardware PWM on Raspberry Pi

## Hardware Requirements

- Raspberry Pi (tested on Pi 4)
- USB or Pi Camera
- 2x Servo motors (pan and tilt)
- Servo mounting hardware

## GPIO Pin Configuration

- Pan Servo: GPIO 12 (hardware PWM)
- Tilt Servo: GPIO 13 (hardware PWM)

## Installation

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-opencv pigpio
   ```

2. **Start pigpio daemon:**
   ```bash
   sudo pigpiod
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- opencv-python==4.10.0.84
- numpy==1.26.4
- pyserial==3.5
- gpiozero==2.0.1
- flask==3.0.0

## Usage

### Basic Usage

```bash
python3 face_tracker_rpi.py
```

### With Command Line Arguments

```bash
python3 face_tracker_rpi.py [camera_index] [serial_port] [baud_rate]
```

**Examples:**
```bash
# Use camera 0, no serial
python3 face_tracker_rpi.py 0

# Use camera 0, with serial port
python3 face_tracker_rpi.py 0 "/dev/ttyUSB0" 115200
```

## Viewing the Camera Stream

Once the program is running, you can view the live camera feed with face detection overlays from any device on your network:

1. **From another device:** Open a web browser and navigate to:
   ```
   http://<raspberry-pi-ip>:5000
   ```
   The IP address will be displayed in the startup messages.

2. **From the Raspberry Pi itself:**
   ```
   http://localhost:5000
   ```

The web interface displays:
- Live video feed
- Face detection bounding boxes (color-coded by face type)
- Face center points
- FPS counter
- Face count

## Configuration

Edit the following constants in `face_tracker_rpi.py` to match your setup:

### Servo Limits
```python
SERVO_PAN_MIN = -0.38    # Minimum pan angle
SERVO_PAN_MAX = 0.38     # Maximum pan angle
SERVO_TILT_MIN = -0.38   # Minimum tilt angle
SERVO_TILT_MAX = 0.38    # Maximum tilt angle
INVERT_Y_SERVO = True    # Set to True if up is negative for your servo
```

### Camera Field of View
```python
HORIZONTAL_FOV_CAM = 50.0  # Camera horizontal FOV in degrees
VERTICAL_FOV_CAM = 50.0    # Camera vertical FOV in degrees
```

### Servo Pins
```python
SERVO_PAN_PIN = 12   # GPIO pin for pan servo
SERVO_TILT_PIN = 13  # GPIO pin for tilt servo
```

## How It Works

1. **Camera Capture** - Captures video frames from the camera
2. **Face Detection** - Uses OpenCV Haar Cascade classifiers to detect frontal and profile faces
3. **Target Selection** - Selects the largest detected face as the tracking target
4. **Coordinate Transformation** - Converts face position to servo angles
5. **Servo Control** - Moves servos to center the face in the frame
6. **Web Streaming** - Streams the annotated video feed via Flask web server

## Face Detection Types

The system detects three types of faces:
- **Frontal faces** (Green bounding box)
- **Left profile faces** (Orange bounding box)
- **Right profile faces** (Light blue bounding box)

## Troubleshooting

### Camera not detected
- Check camera connection
- Try different camera indices (0, 1, 2, etc.)
- Verify camera is not in use by another application

### Servos not responding
- Ensure pigpiod daemon is running: `sudo pigpiod`
- Check GPIO pin connections
- Verify servo power supply

### Cannot access web stream
- Check firewall settings
- Ensure Pi and viewing device are on the same network
- Verify port 5000 is not blocked

### Poor face detection
- Ensure adequate lighting
- Clean camera lens
- Adjust `minNeighbors` parameter in `detect_faces()` method for sensitivity

## Stopping the Program

Press `Ctrl+C` in the terminal to cleanly shut down the tracker and web server.

## License

This project uses OpenCV and pigpio libraries. Please refer to their respective licenses.

## Author

Modified for Raspberry Pi - 2026

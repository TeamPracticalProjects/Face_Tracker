#!/usr/bin/env python3
"""
Face Tracking with OpenCV
Captures video from the laptop camera and tracks faces in real-time.

2026 01 16 removed listing of available cameras to reduce console clutter
           added print of actual camera resolution after initialization
2026 01 18 modified to work with Raspberry Pi and GPIOZero servos

"""
DEBUG = False
FACE_PROFILE_DETECTION = False  # Set to True to detect profile faces (slower but more comprehensive)

import cv2
import sys
import serial
import time
from gpiozero import AngularServo, Device
from gpiozero.pins.pigpio import PiGPIOFactory
from flask import Flask, render_template, Response
import threading
import logging
import os

# Use RPi.GPIO for PWM
Device.pin_factory = PiGPIOFactory()

SERVO_PAN_PIN = 12  # Hardware PWM capable
SERVO_TILT_PIN = 13  # Hardware PWM capable

# set these values to correspond to the field of view of your camera 
# and the mechanism limits. Full range is -1.0 to +1.0
# For a camera with 70 degrees x and y, use about +/-0.14 (70/180 = 0.38)
SERVO_PAN_MIN = -0.38
SERVO_PAN_MAX = 0.38
SERVO_TILT_MIN = -0.38
SERVO_TILT_MAX = 0.38
INVERT_Y_SERVO = True  # Set to True if up is negative for your servo setup

HORIZONTAL_FOV_CAM = 25.0 
VERTICAL_FOV_CAM = 10.0 

# servo setup
servoPan = AngularServo(SERVO_PAN_PIN, min_pulse_width=0.0006, max_pulse_width=0.0024)
servoTilt = AngularServo(SERVO_TILT_PIN, min_pulse_width=0.0006, max_pulse_width=0.0024)

# Flask setup
app = Flask(__name__)
output_frame = None
lock = threading.Lock()
streaming_clients = 0
clients_lock = threading.Lock()


class FaceTracker:
    """Real-time face tracking using OpenCV's Haar Cascade classifier."""
    
    def __init__(self, camera_index=0, serial_port=None, baud_rate=9600):
        """
        Initialize the face tracker.
        
        Args:
            camera_index: Index of the camera to use (default: 0 for built-in webcam)
            serial_port: Serial port path (e.g., '/dev/cu.usbserial-0001')
            baud_rate: Serial baud rate (default: 9600)
        """
        self.camera_index = camera_index
        self.cap = None
        self.face_cascade = None
        self.profile_cascade = None
        self.running = False
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.cam_frame_width = 0
        self.cam_frame_height = 0
        
        # current servo positions, servos should be positioned at center (0.0) at start
        self.servoPanPos = 0.0
        self.servoTiltPos = 0.0

    def initialize(self):
        """Initialize camera and face detection classifier."""
        print("Initializing face tracker...")
        
        # Load the pre-trained Haar Cascade classifiers for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            print("Error: Could not load frontal face cascade classifier")
            return False
        
        # Load profile face detector
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        if self.profile_cascade.empty():
            print("Error: Could not load profile face cascade classifier")
            return False
        
        print("Loaded frontal and profile face detectors")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}")
            return False
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
        
        # Get actual frame dimensions
        self.cam_frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {self.cam_frame_width}x{self.cam_frame_height}")
        
        # Initialize serial port if specified
        if self.serial_port:
            try:
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                time.sleep(2)  # Wait for serial connection to initialize
                print(f"Serial port opened: {self.serial_port} at {self.baud_rate} baud")
            except Exception as e:
                print(f"Warning: Could not open serial port {self.serial_port}: {e}")
                self.ser = None
        
        print("Face tracker initialized successfully!")
        return True
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame (frontal and profile).
        
        Args:
            frame: Input image frame
            
        Returns:
            List of (x, y, w, h, type) tuples for detected faces
        """
        # Convert to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        frontal_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Combine all detections with labels
        all_faces = []
        
        # Add frontal faces
        for (x, y, w, h) in frontal_faces:
            all_faces.append((x, y, w, h, 'frontal'))
        
        # Optionally detect profile faces (slower but detects more faces)
        if FACE_PROFILE_DETECTION:
            # Detect profile faces (left-facing)
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detect profile faces (right-facing) by flipping the image
            gray_flipped = cv2.flip(gray, 1)
            profile_faces_flipped = self.profile_cascade.detectMultiScale(
                gray_flipped,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add left profile faces
            for (x, y, w, h) in profile_faces:
                all_faces.append((x, y, w, h, 'profile-left'))
            
            # Add right profile faces (adjust coordinates back from flipped image)
            frame_width = gray.shape[1]
            for (x, y, w, h) in profile_faces_flipped:
                x_corrected = frame_width - x - w
                all_faces.append((x_corrected, y, w, h, 'profile-right'))
        
        # Remove overlapping detections
        all_faces = self._remove_overlaps(all_faces)
        
        return all_faces
    
    def _remove_overlaps(self, faces, overlap_threshold=0.3):
        """
        Remove overlapping face detections.
        
        Args:
            faces: List of (x, y, w, h, type) tuples
            overlap_threshold: Minimum overlap ratio to consider as duplicate
            
        Returns:
            Filtered list of faces
        """
        if len(faces) == 0:
            return faces
        
        filtered = []
        
        for i, (x1, y1, w1, h1, type1) in enumerate(faces):
            overlap = False
            
            for j, (x2, y2, w2, h2, type2) in enumerate(faces):
                if i != j:
                    # Calculate intersection area
                    xi1 = max(x1, x2)
                    yi1 = max(y1, y2)
                    xi2 = min(x1 + w1, x2 + w2)
                    yi2 = min(y1 + h1, y2 + h2)
                    
                    if xi2 > xi1 and yi2 > yi1:
                        intersection = (xi2 - xi1) * (yi2 - yi1)
                        area1 = w1 * h1
                        area2 = w2 * h2
                        
                        # Check if significant overlap
                        if intersection / min(area1, area2) > overlap_threshold:
                            # Keep the larger detection
                            if area1 < area2:
                                overlap = True
                                break
            
            if not overlap:
                filtered.append((x1, y1, w1, h1, type1))
        
        return filtered
    
    def send_to_serial(self, faces):
        """
        Send the center coordinates of the largest face to serial port.
        Coordinates are transformed to have (0,0) at center of frame.
        
        Args:
            faces: List of detected faces
        """
    
        if len(faces) == 0:
            return
        
        # Find the largest face by area
        largest_face = max(faces, key=lambda f: f[2] * f[3])  # w * h
        x, y, w, h, face_type = largest_face
        
        # Point of Interest (center of the largest face)
        poi_x_cam = x + w // 2
        poi_y_cam = y + h // 2

        print(f"Largest face center: ({poi_x_cam}, {poi_y_cam})")
        
        # Transform to center-origin coordinate system
        # (0,0) at center of frame, positive X to left, positive Y up
        poi_x_cam_center_origin = poi_x_cam - (self.cam_frame_width // 2) 
        poi_y_cam_center_origin = (poi_y_cam - (self.cam_frame_height // 2)) * -1  # Camera has Inverted Y
        if DEBUG:
            print(f"Transformed to center origin: ({poi_x_cam_center_origin}, {poi_y_cam_center_origin})")
        """
        # Scale transformed_x to servo pan range
        # Map from [-frame_width/2, frame_width/2] to [SERVO_PAN_MIN, SERVO_PAN_MAX]
        max_x = self.frame_width // 2
        if max_x > 0:
            scaled_x = (transformed_x / max_x) * ((SERVO_PAN_MAX - SERVO_PAN_MIN) / 2)
            scaled_x = max(SERVO_PAN_MIN, min(SERVO_PAN_MAX, scaled_x))
        else:
            scaled_x = 0
        
        # Scale transformed_y to servo tilt range
        max_y = self.frame_height // 2
        if max_y > 0:
            scaled_y = (transformed_y / max_y) * ((SERVO_TILT_MAX - SERVO_TILT_MIN) / 2)
            scaled_y = max(SERVO_TILT_MIN, min(SERVO_TILT_MAX, scaled_y))
        else:
            scaled_y = 0
        print(f"Scaled to servo range: ({scaled_x:.3f}, {scaled_y:.3f})")
        """
        # scale center origin to servo movement increments in FOV units
        scaled_x = poi_x_cam_center_origin  * (HORIZONTAL_FOV_CAM / self.cam_frame_width)
        scaled_y = poi_y_cam_center_origin  * (VERTICAL_FOV_CAM / self.cam_frame_height)
        if DEBUG:
            print(f"Scaled to FOV increments: ({scaled_x:.3f}, {scaled_y:.3f})")

        # convert increments to servo position changes (-1.0 to +1.0)
        scaled_x_servo = scaled_x / self.cam_frame_width
        scaled_y_servo = scaled_y / self.cam_frame_height
        if DEBUG:
            print(f"Scaled to servo position changes: ({scaled_x_servo:.3f}, {scaled_y_servo:.3f})")

        if DEBUG:
            print(f"Current servo positions before update: ({self.servoPanPos:.3f}, {self.servoTiltPos:.3f})")
        # Move the position relative to current position
        self.servoPanPos -= scaled_x_servo
        if INVERT_Y_SERVO:
            scaled_y_servo = -scaled_y_servo
        self.servoTiltPos += scaled_y_servo
        if DEBUG:
            print(f"Updated servo positions before clamp: ({self.servoPanPos:.3f}, {self.servoTiltPos:.3f})")

        # Clamp to servo limits
        self.servoPanPos = max(-.8, min(.8, self.servoPanPos))
        self.servoTiltPos = max(-.8, min(.8, self.servoTiltPos))   
        if DEBUG:
            print(f"Clamped servo positions: ({self.servoPanPos:.3f}, {self.servoTiltPos:.3f})")

        # TESTING
        # self.servoPanPos = 0.0

        if DEBUG:
            print(f"Sent to servos: {self.servoPanPos:.3f}, {self.servoTiltPos:.3f}")
        # Move Servos
        try:
            servoPan.value = self.servoPanPos
            servoTilt.value = self.servoTiltPos
            time.sleep(0.2) # wait for servo to repond before sending new data
        except Exception as e:
            print(f"servo write error: {e}")
    
    def draw_faces(self, frame, faces):
        """
        Draw bounding boxes and labels around detected faces.
        
        Args:
            frame: Image frame to draw on
            faces: List of (x, y, w, h, type) tuples for detected faces
        """
        for i, (x, y, w, h, face_type) in enumerate(faces):
            # Print coordinates
            #print(f"Face {i + 1} ({face_type}): x={x}, y={y}, width={w}, height={h}, "
             #     f"top-left=({x},{y}), bottom-right=({x+w},{y+h})")
            
            # Choose color based on face type
            if face_type == 'frontal':
                color = (0, 255, 0)  # Green for frontal
            elif face_type == 'profile-left':
                color = (255, 165, 0)  # Orange for left profile
            else:  # profile-right
                color = (0, 165, 255)  # Light blue for right profile
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw face center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Add label
            label = f"Face {i + 1} ({face_type})"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def add_info_overlay(self, frame, faces, fps):
        """
        Add information overlay to the frame.
        
        Args:
            frame: Image frame to draw on
            faces: List of detected faces
            fps: Current frames per second
        """
        # Add face count
        text = f"Faces detected: {len(faces)}"
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions
        instructions = "Press 'q' to quit"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw center crosshairs
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        color = (0, 255, 255)  # Yellow color
        
        # Draw vertical dashed line
        dash_length = 10
        gap_length = 5
        for y in range(0, height, dash_length + gap_length):
            y_end = min(y + dash_length, height)
            cv2.line(frame, (center_x, y), (center_x, y_end), color, 1)
        
        # Draw horizontal dashed line
        for x in range(0, width, dash_length + gap_length):
            x_end = min(x + dash_length, width)
            cv2.line(frame, (x, center_y), (x_end, center_y), color, 1)
    
    def run(self):
        """Run the face tracking loop."""
        if not self.initialize():
            return
        
        self.running = True
        print("\nFace tracking started!")
        print("Press 'q' to quit\n")
        
        # For FPS calculation
        fps = 0
        frame_count = 0
        import time
        start_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Send first face coordinates to serial port
                self.send_to_serial(faces)
                
                # Draw faces
                self.draw_faces(frame, faces)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                
                # Add info overlay
                self.add_info_overlay(frame, faces, fps)
                
                # Store frame for Flask streaming (only if clients are connected)
                global output_frame, lock, streaming_clients, clients_lock
                with clients_lock:
                    has_clients = streaming_clients > 0
                
                if has_clients:
                    with lock:
                        output_frame = frame.copy()
                
                # Check for quit key (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        if self.ser is not None:
            self.ser.close()
            print("Serial port closed")
        
        cv2.destroyAllWindows()
        print("\nFace tracker stopped")


def list_available_cameras(max_cameras=32):
    """
    List all available cameras.
    
    Args:
        max_cameras: Maximum number of camera indices to check
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def generate_frames():
    """Generate frames for Flask streaming."""
    global output_frame, lock, streaming_clients, clients_lock
    
    # Track this client
    with clients_lock:
        streaming_clients += 1
    
    try:
        while True:
            with lock:
                if output_frame is None:
                    continue
                
                # Encode frame as JPEG with reduced quality (60% instead of default 95%)
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame, encode_params)
                
                if not flag:
                    continue
            
            # Yield frame in MJPEG format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
    finally:
        # Untrack this client when they disconnect
        with clients_lock:
            streaming_clients -= 1


@app.route('/')
def index():
    """Video streaming home page."""
    return """<!DOCTYPE html>
<html>
<head>
    <title>Face Tracker Stream</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        h1 {
            color: #4CAF50;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        .info {
            margin-top: 20px;
            color: #cccccc;
        }
    </style>
</head>
<body>
    <h1>Face Tracker - Live Stream</h1>
    <img src="/video_feed" alt="Video Stream">
    <div class="info">
        <p>Press Ctrl+C in the terminal to stop the tracker</p>
    </div>
</body>
</html>"""


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    """Main entry point."""
    print("=" * 60)
    print("OpenCV Face Tracker")
    print("=" * 60)
    
    # Get and display IP address for web streaming
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
    except Exception:
        ip_address = "localhost"
    
    print("\n" + "=" * 60)
    print("CAMERA STREAM WILL BE AVAILABLE AT:")
    print(f"  http://{ip_address}:5000")
    if ip_address != "localhost":
        print(f"  http://localhost:5000 (from this device)")
    print("=" * 60)
    
    # List available cameras
    available_cameras = list_available_cameras()
    print(f"\nAvailable cameras: {available_cameras}")

    # Test servos
    testDelaySeconds = 1
    print("\nTesting servos...")
    print("Panning servos to FOV left")
    servoPan.value = SERVO_PAN_MAX
    servoTilt.value = 0
    time.sleep(testDelaySeconds)
    print("Panning servos to FOV right")
    servoPan.value = SERVO_PAN_MIN
    time.sleep(testDelaySeconds)    
    print("Tilting servos to FOV up")
    servoPan.value = 0
    if INVERT_Y_SERVO:
        servoTilt.value = SERVO_TILT_MIN
    else:
        servoTilt.value = SERVO_TILT_MAX
    time.sleep(testDelaySeconds)
    print("Panning servos to FOV down")
    servoPan.value = 0
    if INVERT_Y_SERVO:
        servoTilt.value = SERVO_TILT_MAX
    else:       
        servoTilt.value = SERVO_TILT_MIN
    time.sleep(testDelaySeconds)
    servoPan.value = 0
    servoTilt.value = 0
    print("Servos test complete.")

    # Parse command line arguments
    camera_index = 0
    serial_port = "COM5"  # Use the port that your Arduino is on
    baud_rate = 115200
    
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
            print(f"Using camera index: {camera_index}")
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}, using default (0)")
    
    if len(sys.argv) > 2:
        serial_port = sys.argv[2]
        print(f"Serial port: {serial_port}")
    
    if len(sys.argv) > 3:
        try:
            baud_rate = int(sys.argv[3])
            print(f"Baud rate: {baud_rate}")
        except ValueError:
            print(f"Invalid baud rate: {sys.argv[3]}, using default (9600)")
    
    print("\nUsage: python face_tracker.py [camera_index] [serial_port] [baud_rate]")
    print("Example: python face_tracker.py 0 \"COM5\" 9600")
    print("Example: python face_tracker.py 1 \"/dev/cu.usbserial-0001\" 115200")
    print()
    
    # Create and run tracker
    tracker = FaceTracker(camera_index=camera_index, serial_port=serial_port, baud_rate=baud_rate)
    
    # Run tracker in a separate thread
    tracker_thread = threading.Thread(target=tracker.run, daemon=True)
    tracker_thread.start()
    
    # Suppress Flask startup messages
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Start Flask server
    print("\n" + "="*60)
    print("Starting Flask web server...")
    print(f"View the stream at: http://{ip_address}:5000")
    if ip_address != "localhost":
        print(f"  Or from this device: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        tracker.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

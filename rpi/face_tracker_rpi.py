#!/usr/bin/env python3
"""
Face Tracking with OpenCV
Captures video from the laptop camera and tracks faces in real-time.

2026 01 16 removed listing of available cameras to reduce console clutter
           added print of actual camera resolution after initialization
2026 01 18 modified to work with Raspberry Pi and GPIOZero servos
2026 01 21 added profile face detection option to increase speed
           reduced jpeg quality for web streaming to reduce bandwidth
           changed servo movement to fixed increments for smoother movement
              while also reducing sleep time to improve responsiveness

"""
DEBUG = False
FACE_PROFILE_DETECTION = False  # Set to True to detect profile faces (slower but more comprehensive)
DETECT_EVERY_X_FRAMES = 3  # Detect every x frames for performance

import cv2
import sys
import serial
import time
import argparse
from gpiozero import AngularServo, Device
from gpiozero.pins.pigpio import PiGPIOFactory
from flask import Flask, render_template, Response
import threading
import logging
import os
import numpy as np
import subprocess
import io

# Try to import picamera2 for Pi camera support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    print("Picamera2 library is available")
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Note: picamera2 not available, will try libcamera-still/vid")



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
        self.use_picamera2 = False
        self.picam2 = None
        
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
        
        # Initialize video capture with Raspberry Pi camera support
        self.cap = self._initialize_camera(self.camera_index)
        
        if self.cap is None or not self.cap.isOpened():
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
    
    def _initialize_camera(self, camera_index):
        """Initialize camera with support for both USB and Raspberry Pi CSI cameras."""
        print(f"Attempting to open camera {camera_index}...")
        print(f"Picamera2 available: {PICAMERA2_AVAILABLE}")
        
        # Try 0: For camera index 0-2 on Raspberry Pi, try picamera2 first if available
        if camera_index <= 2 and PICAMERA2_AVAILABLE:
            print("Trying Picamera2 library for Raspberry Pi camera...")
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                print("Starting Picamera2...")
                self.picam2.start()
                time.sleep(2)  # Let camera warm up
                
                # Test capture
                print("Testing frame capture...")
                frame = self.picam2.capture_array()
                if frame is not None and frame.size > 0:
                    print(f"Successfully initialized Picamera2!")
                    print(f"Frame size: {frame.shape}")
                    self.use_picamera2 = True
                    # Create a dummy VideoCapture object for compatibility
                    class Picamera2Wrapper:
                        def __init__(self, picam2):
                            self.picam2 = picam2
                        
                        def isOpened(self):
                            return True
                        
                        def read(self):
                            frame = self.picam2.capture_array()
                            # Picamera2 returns RGB, but OpenCV expects BGR
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            return True, frame
                        
                        def release(self):
                            if self.picam2:
                                self.picam2.stop()
                        
                        def get(self, prop):
                            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                                return 640
                            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                                return 480
                            return 0
                        
                        def set(self, prop, value):
                            # Ignore set calls for picamera2
                            return True
                    
                    return Picamera2Wrapper(self.picam2)
                else:
                    print("Picamera2 capture returned invalid frame")
            except Exception as e:
                print(f"Picamera2 failed: {e}")
                import traceback
                traceback.print_exc()
                if self.picam2:
                    try:
                        self.picam2.stop()
                    except:
                        pass
                    self.picam2 = None
        elif camera_index <= 2:
            print("Picamera2 not available, trying other methods...")
        
        # Try 1: Try /dev/videoX explicitly with V4L2 first (most reliable for Pi camera)
        device_path = f"/dev/video{camera_index}"
        print(f"Trying {device_path} with V4L2 backend...")
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Successfully opened {device_path} with V4L2")
            # For Pi CSI camera, we need to set format properties in specific order
            # Set pixel format first
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Set other properties
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Set convert RGB flag - important for some Pi cameras
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            
            print(f"Configured camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print("Waiting for camera to initialize...")
            time.sleep(2)  # Longer wait for Pi camera
            
            # Flush any initial frames and try reading
            print("Flushing initial frames...")
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Successfully reading frames from {device_path}")
                    print(f"Frame size: {frame.shape}, dtype: {frame.dtype}")
                    return cap
                time.sleep(0.1)
            
            print(f"Could not read valid frames after multiple attempts")
            cap.release()
        else:
            cap.release()
        
        # Try 2: Try with different backend initialization order
        print(f"Trying {device_path} with alternate V4L2 configuration...")
        cap = cv2.VideoCapture()
        cap.open(device_path, cv2.CAP_V4L2)
        if cap.isOpened():
            # Try with minimal configuration
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            time.sleep(2)
            
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Successfully reading with alternate config")
                    print(f"Frame size: {frame.shape}")
                    return cap
                time.sleep(0.1)
            cap.release()
        
        # Try 3: GStreamer pipeline for Pi camera with explicit format
        print(f"Trying GStreamer pipeline for {device_path}...")
        gstreamer_pipeline = (
            f"v4l2src device={device_path} ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink"
        )
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("Successfully opened camera with GStreamer pipeline")
            time.sleep(1)
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Successfully reading frames via GStreamer")
                    print(f"Frame size: {frame.shape}")
                    return cap
                time.sleep(0.2)
            cap.release()
        
        # Try 4: V4L2 backend with camera index
        print(f"Trying camera index {camera_index} with V4L2 backend...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Successfully opened camera {camera_index} with V4L2 backend")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Frame size: {frame.shape}")
                    return cap
                time.sleep(0.2)
            cap.release()
        else:
            cap.release()
        
        # Try 3: Direct index (works for USB cameras)
        print(f"Trying camera index {camera_index} with default backend...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Successfully opened camera {camera_index} with default backend")
            ret, _ = cap.read()
            if ret:
                return cap
            else:
                print("Could open device but cannot read frames, trying next method...")
                cap.release()
        else:
            cap.release()
        
        # Try 4: For Raspberry Pi camera (index 0), try GStreamer pipeline
        if camera_index == 0:
            print("Trying Raspberry Pi camera via GStreamer with libcamerasrc...")
            gstreamer_pipeline = (
                "libcamerasrc ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
                "videoconvert ! appsink"
            )
            cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("Successfully opened Raspberry Pi camera with GStreamer")
                ret, _ = cap.read()
                if ret:
                    return cap
                else:
                    print("Could open device but cannot read frames, trying next method...")
                    cap.release()
            else:
                cap.release()
            
            # Try 5: Alternative GStreamer pipeline
            print("Trying GStreamer with v4l2src...")
            gstreamer_pipeline_v4l2 = (
                "v4l2src device=/dev/video0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
                "videoconvert ! appsink"
            )
            cap = cv2.VideoCapture(gstreamer_pipeline_v4l2, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("Successfully opened camera with GStreamer v4l2src")
                ret, _ = cap.read()
                if ret:
                    return cap
                else:
                    print("Could open device but cannot read frames")
                    cap.release()
            else:
                cap.release()
        
        print(f"Failed to open camera {camera_index} with any method")
        print("\nPlease check:")
        print("  1. Is the camera connected properly?")
        print("  2. Run: ls -l /dev/video*  (to see available video devices)")
        print("  3. Run: v4l2-ctl --list-devices  (to see camera details)")
        print("  4. Install v4l-utils if needed: sudo apt-get install v4l-utils")
        return None
    
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

        #print(f"Largest face center: ({poi_x_cam}, {poi_y_cam})")
        
        # Transform to center-origin coordinate system
        # (0,0) at center of frame, positive X to left, positive Y up
        poi_x_cam_center_origin = poi_x_cam - (self.cam_frame_width // 2) 
        poi_y_cam_center_origin = (poi_y_cam - (self.cam_frame_height // 2)) * -1  # Camera has Inverted Y
        if DEBUG:
            print(f"Transformed to center origin: ({poi_x_cam_center_origin}, {poi_y_cam_center_origin})")
        
        # scale center origin to servo movement increments in FOV units
        scaled_x = poi_x_cam_center_origin  * (HORIZONTAL_FOV_CAM / self.cam_frame_width)
        scaled_y = poi_y_cam_center_origin  * (VERTICAL_FOV_CAM / self.cam_frame_height)
        if DEBUG:
            print(f"Scaled to FOV increments: ({scaled_x:.3f}, {scaled_y:.3f})")

        # convert increments to servo position changes (-1.0 to +1.0)
        amount_to_move_x = scaled_x / self.cam_frame_width
        amount_to_move_y = scaled_y / self.cam_frame_height
        if DEBUG:
            print(f"Scaled to servo position changes: ({amount_to_move_x:.3f}, {amount_to_move_y:.3f})")

        # TESTING - limit to fixed increment for smoother movement
        # reduced the sleep at the end of this function
        # also commented out the print of face detection coordinates above
        increment = 0.005
        no_movement_threshold = 0.002  # servo distance from center before moving

        # Only move if face is not centered (outside the threshold)
        if abs(poi_x_cam_center_origin) > no_movement_threshold:
            if poi_x_cam_center_origin > 0:
                amount_to_move_x = increment
            else:
                amount_to_move_x = -increment
        else:
            amount_to_move_x = 0  # Don't move if centered

        if abs(poi_y_cam_center_origin) > no_movement_threshold:
            if poi_y_cam_center_origin > 0:
                amount_to_move_y = increment
            else:
                amount_to_move_y = -increment
        else:
            amount_to_move_y = 0  # Don't move if centered

        if DEBUG:
            print(f"Current servo positions before update: ({self.servoPanPos:.3f}, {self.servoTiltPos:.3f})")
        # Move the position relative to current position
        self.servoPanPos -= amount_to_move_x
        if INVERT_Y_SERVO:
            amount_to_move_y = -amount_to_move_y
        self.servoTiltPos += amount_to_move_y
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
        except Exception as e:
            print(f"servo write error: {e}")

        #time.sleep(0.001) # wait for servo to repond before sending new data
    
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
            # Keep the last detection results so we can reuse them on skipped frames
            faces = []
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()

                if not ret:
                    print("Error: Failed to capture frame")
                    break

                '''
                # Scale frame to width of 320 for better performance
                height, width = frame.shape[:2]
                target_width = 320
                scale_factor = target_width / width
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (target_width, new_height))
                '''
                # Perform face detection only on every 3rd frame to reduce CPU
                if frame_count % DETECT_EVERY_X_FRAMES == 0:
                    faces = self.detect_faces(frame)

                # Send first face coordinates to serial port (uses most recent detections)
                self.send_to_serial(faces)

                # Draw faces (uses most recent detections)
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
        
        if self.use_picamera2 and self.picam2 is not None:
            try:
                self.picam2.stop()
                print("Picamera2 stopped")
            except:
                pass
        
        if self.libcamera_process is not None:
            try:
                self.libcamera_process.terminate()
                self.libcamera_process.wait(timeout=2)
                print("libcamera process terminated")
            except:
                pass
        
        if self.cap is not None:
            self.cap.release()
        
        if self.ser is not None:
            self.ser.close()
            print("Serial port closed")
        
        cv2.destroyAllWindows()
        print("\nFace tracker stopped")


def list_available_cameras(max_cameras=32, detailed=False):
    """
    List all available cameras.
    
    Args:
        max_cameras: Maximum number of camera indices to check
        detailed: If True, also show camera resolution and backend
        
    Returns:
        List of available camera indices
    """
    available = []
    print(f"\nScanning for cameras (checking indices 0-{max_cameras-1})...")
    print("-" * 60)
    
    # Check for Raspberry Pi camera first
    print("Checking for Raspberry Pi CSI camera...")
    # Try GStreamer pipeline for Pi camera
    gstreamer_pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Raspberry Pi Camera: {width}x{height} (GStreamer/libcamera)")
        available.append(0)  # Pi camera is typically camera 0
        cap.release()
    else:
        cap.release()
        # Try /dev/video0 with V4L2
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera at /dev/video0: {width}x{height} (V4L2)")
            if 0 not in available:
                available.append(0)
            cap.release()
    
    print("\nChecking USB cameras...")
    for i in range(max_cameras):
        if i in available:
            continue  # Already found this camera
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            if detailed:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = cap.getBackendName()
                print(f"Camera {i}: {width}x{height} (backend: {backend})")
            else:
                print(f"Camera {i}: Available")
            cap.release()
    
    print("-" * 60)
    if available:
        print(f"\nFound {len(available)} camera(s): {sorted(available)}")
    else:
        print("\nNo cameras found.")
        print("\nTroubleshooting tips for Raspberry Pi Camera:")
        print("  1. Check physical connection - ensure ribbon cable is properly inserted")
        print("  2. Test camera with: libcamera-hello")
        print("     (If this works, the camera hardware is functional)")
        print("  3. Check if camera is detected: vcgencmd get_camera")
        print("  4. Install required packages:")
        print("     sudo apt-get update")
        print("     sudo apt-get install -y python3-opencv")
        print("     sudo apt-get install -y libcamera-apps")
        print("  5. For GStreamer support:")
        print("     sudo apt-get install -y gstreamer1.0-tools")
        print("     sudo apt-get install -y gstreamer1.0-plugins-base")
        print("     sudo apt-get install -y gstreamer1.0-plugins-good")
        print("  6. Reboot after installing packages")
        print("\n  Note: Modern Raspberry Pi OS uses libcamera (no raspi-config needed)")
    
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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Face Tracking with OpenCV and Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s --list-cameras               List available cameras and exit
  %(prog)s -l                            List available cameras and exit
  %(prog)s 0 COM5 115200                Use camera 0 with serial port
  %(prog)s 1 /dev/cu.usbserial-0001     Use camera 1 with serial port'''
    )
    parser.add_argument(
        '-l', '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )
    parser.add_argument(
        'camera_index',
        nargs='?',
        type=int,
        default=0,
        help='Camera index to use (default: 0)'
    )
    parser.add_argument(
        'serial_port',
        nargs='?',
        default='COM5',
        help='Serial port for Arduino (default: COM5)'
    )
    parser.add_argument(
        'baud_rate',
        nargs='?',
        type=int,
        default=115200,
        help='Serial baud rate (default: 115200)'
    )
    
    args = parser.parse_args()
    
    # Handle list-cameras option
    if args.list_cameras:
        list_available_cameras(max_cameras=32, detailed=True)
        return 0

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

    # Use parsed arguments
    camera_index = args.camera_index
    serial_port = args.serial_port
    baud_rate = args.baud_rate
    
    print(f"\nUsing camera index: {camera_index}")
    print(f"Serial port: {serial_port}")
    print(f"Baud rate: {baud_rate}")
    
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

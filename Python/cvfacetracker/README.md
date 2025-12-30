# OpenCV Test Project

A Python project for testing OpenCV functionality and experimenting with computer vision operations.

## Features

- Basic OpenCV installation verification
- Image creation and manipulation tests
- Shape drawing demonstrations
- Image processing operations (grayscale conversion, blur, edge detection)

## Requirements

- Python 3.8 or higher
- OpenCV (opencv-python)
- NumPy

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the basic test script:
```bash
python test_opencv.py
```

This will execute all OpenCV tests and create a sample output image (`test_output.png`) demonstrating basic drawing operations.

### Run the face tracking program:
```bash
python face_tracker.py
```

This will:
- Open your laptop's camera
- Detect and track faces in real-time
- Display bounding boxes around detected faces
- Show FPS and face count
- Output to the serial port the center coordinates of a box bounding the largest face detected
- Press 'q' or ESC to quit

To use a different camera (if you have multiple cameras):
```bash
python face_tracker.py 1  # Use camera index 1 serial-port
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── test_opencv.py          # Basic OpenCV functionality tests
├── face_tracker.py         # Real-time face tracking from camera
└── .github/
    └── copilot-instructions.md
```

## What the Tests Cover

### test_opencv.py
1. **Installation Test**: Verifies OpenCV is installed and prints version information
2. **Image Creation**: Creates blank and colored images
3. **Shape Drawing**: Demonstrates rectangles, circles, lines, and text
4. **Image Operations**: Shows grayscale conversion, blurring, and edge detection

### face_tracker.py
1. **Camera Access**: Opens and configures laptop webcam
2. **Face Detection**: Uses Haar Cascade classifier for real-time face detection
3. **Visual Tracking**: Draws bounding boxes and center points on detected faces
4. **Performance Monitoring**: Displays FPS and face count
5. **Interactive Control**: Quit with 'q' or ESC key

## Next Steps

- Add more complex image processing tests
- Implement video capture and processing
- Add face detection examples
- Create image transformation demonstrations

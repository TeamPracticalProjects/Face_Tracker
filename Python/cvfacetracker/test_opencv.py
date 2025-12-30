#!/usr/bin/env python3
"""
OpenCV Test Script
Tests basic OpenCV functionality including image loading, display, and basic operations.
"""

import cv2
import numpy as np
import sys


def test_opencv_installation():
    """Test if OpenCV is properly installed and print version."""
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    return True


def test_create_image():
    """Test creating a simple image with OpenCV."""
    print("\n--- Testing Image Creation ---")
    
    # Create a blank image (black)
    height, width = 480, 640
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)
    print(f"Created blank image: {blank_image.shape}")
    
    # Create an image with color
    blue_image = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
    print(f"Created blue image: {blue_image.shape}")
    
    return True


def test_draw_shapes():
    """Test drawing basic shapes on an image."""
    print("\n--- Testing Shape Drawing ---")
    
    # Create a blank white canvas
    img = np.full((480, 640, 3), (255, 255, 255), dtype=np.uint8)
    
    # Draw a rectangle
    cv2.rectangle(img, (100, 100), (300, 200), (0, 255, 0), 2)
    print("Drew rectangle")
    
    # Draw a circle
    cv2.circle(img, (450, 150), 50, (0, 0, 255), -1)
    print("Drew circle")
    
    # Draw a line
    cv2.line(img, (100, 300), (540, 300), (255, 0, 0), 3)
    print("Drew line")
    
    # Add text
    cv2.putText(img, "OpenCV Test", (200, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print("Added text")
    
    # Save the test image
    cv2.imwrite("test_output.png", img)
    print("Saved test image to test_output.png")
    
    return True


def test_image_operations():
    """Test basic image operations."""
    print("\n--- Testing Image Operations ---")
    
    # Create a sample gradient image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[i, :] = [i, i, i]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Converted to grayscale: {gray.shape}")
    
    # Blur the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    print(f"Applied Gaussian blur: {blurred.shape}")
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    print(f"Applied Canny edge detection: {edges.shape}")
    
    return True


def main():
    """Run all OpenCV tests."""
    print("=" * 50)
    print("OpenCV Test Suite")
    print("=" * 50)
    
    try:
        test_opencv_installation()
        test_create_image()
        test_draw_shapes()
        test_image_operations()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

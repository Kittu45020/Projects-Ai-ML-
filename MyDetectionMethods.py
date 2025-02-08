# This program captures live video from a USB camera to detect and measure objects in real-time. 
# It uses ArUco markers for scaling and OpenCV for edge detection, contour analysis, and object classification. 
# Detected objects are identified based on their dimensions and annotated in the video feed. 
# The program is designed for applications involving real-time object measurement and detection.

#--program Starts here--
# Import necessary libraries for image processing, numerical operations, and timing
import cv2
import numpy as np
import time

# Purpose: Define a class for custom image detection methods
# This class encapsulates reusable methods for processing images, such as edge detection.
class MyDetectionMethods:
    def __init__(self):
        pass

    # Purpose: Apply the Canny edge detection algorithm
    # This method detects edges in an input image, simplifying contour analysis.
    def apply_canny(self, image, threshold1=50, threshold2=150):
        """
        Detect edges in the image using the Canny algorithm.
        Converts the image to grayscale if it's in color.
        Parameters:
        - image: Input image.
        - threshold1: Lower threshold for the hysteresis procedure.
        - threshold2: Upper threshold for the hysteresis procedure.
        Returns:
        - edges: The edge-detected image.
        """
        if len(image.shape) == 3:  # Check if the image is in color (BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, threshold1, threshold2)
        return edges

# Purpose: Initialize the detection class and set up the camera
detector = MyDetectionMethods()

# Purpose: Configure the USB camera for live video feed
camera = cv2.VideoCapture(1)  # Camera index 1 (may vary depending on system setup)
time.sleep(0.5)  # Short delay to ensure the camera initializes properly

# Purpose: Validate camera access before proceeding
if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Purpose: Prepare for detecting ArUco markers and object measurement
# Load the predefined dictionary for ArUco markers (5x5 marker grid with 100 markers)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Initialize ArUco marker detector parameters (can be tuned for better performance)
parameters = cv2.aruco.DetectorParameters()

# Define the known physical size of the marker in centimeters (used for scaling)
marker_size_cm = 10.0

print("Press 'q' to quit the live video feed.\n")

# Purpose: Main loop for live video feed and object detection
while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Purpose: Detect ArUco markers in the current frame
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:  # If markers are detected
        # Draw detected markers and their IDs on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Extract the corners of the first detected marker (used for scaling)
        marker_corners = corners[0][0]

        # Calculate the average size of the marker in pixels (width and height)
        pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
        pixel_height = np.linalg.norm(marker_corners[1] - marker_corners[2])
        average_pixel_size = (pixel_width + pixel_height) / 2

        # Compute the pixel-to-centimeter ratio using the known marker size
        pixel_to_cm_ratio = average_pixel_size / marker_size_cm

        # Convert the frame to grayscale for edge detection and contour analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection to find object boundaries
        edges = detector.apply_canny(gray)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each detected contour
        for contour in contours:
            # Filter out small contours to reduce noise
            if cv2.contourArea(contour) < 100:
                continue

            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Convert the bounding box dimensions from pixels to centimeters
            object_width_cm = w / pixel_to_cm_ratio
            object_height_cm = h / pixel_to_cm_ratio

            # Classify objects based on their dimensions (in cm)
            if 1.0 <= object_width_cm <= 1.3 and 3.7 <= object_height_cm <= 4.5:
                print(f"Battery Detected - Width: {object_width_cm:.1f} cm, Height: {object_height_cm:.1f} cm")
                color = (0, 0, 255)  # Red color for battery
                label = f"Battery - {object_width_cm:.1f} x {object_height_cm:.1f} cm"
            elif 8.0 <= object_width_cm <= 11.0 and 5.0 <= object_height_cm <= 7.5:
                print(f"Credit Card Detected - Width: {object_width_cm:.1f} cm, Height: {object_height_cm:.1f} cm")
                color = (255, 0, 0)  # Blue color for credit card
                label = f"Credit Card - {object_width_cm:.1f} x {object_height_cm:.1f} cm"
            else:
                continue  # Skip contours that don't match predefined object dimensions

            # Draw a bounding box around the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Add a label with the object's dimensions above the bounding box
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Purpose: Display the annotated video frame
    cv2.imshow("Object Measurement", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Purpose: Release camera resources and close display windows
camera.release()
cv2.destroyAllWindows()

# Image Flow: The input image is captured in real-time from a USB camera and undergoes multiple processing stages.
# These stages include ArUco marker detection to calculate the pixel-to-centimeter scale for accurate measurements,
# grayscale conversion for simplifying image data, and edge detection to highlight object boundaries.
# The contours of the objects are extracted for further analysis, such as size classification and object detection.

# Final Output: The processed frame is annotated with bounding boxes, labels, and dimensions for each detected object.
# These annotations help identify object characteristics, such as size and category (e.g., battery or credit card).
# The results are displayed on the screen in real-time, enabling users to visualize and verify object detection outputs.








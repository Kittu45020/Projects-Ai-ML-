# Purpose of the Code:
# This program is designed to analyze needle-like objects in a live video feed captured through a camera.
# It performs the following tasks:
# 1. Detects ArUco markers to calculate a pixel-to-millimeter scale for accurate size measurements.
# 2. Detects objects using edge detection and contour analysis, focusing on needle-like shapes.
# 3. Classifies detected objects based on predefined size categories (e.g., small, large, or defect).
# 4. Identifies the dominant color of each object and checks for potential tip overlaps between needles.
# 5. Annotates the live video feed with detailed information such as size, color, angle, and overlap status.
# This program is suitable for applications requiring real-time needle analysis, such as in industrial or medical environments.

#The Code program starts from here
import cv2
import numpy as np

# Purpose: Define a class for needle detection and scale calibration
class NeedleDetector:
    def __init__(self):  # Constructor to initialize attributes
        self.pixel_to_mm = None  # Scale factor for converting pixels to millimeters

    # Purpose: Calculate pixel-to-millimeter scale using an ArUco marker
    # This method detects an ArUco marker, measures its side length in pixels,
    # and calculates the scale factor based on its known physical size.
    def calculate_scale_from_aruco(self, frame, aruco_length_mm=75):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)  # Predefined dictionary for ArUco markers
        parameters = cv2.aruco.DetectorParameters()  # Detection parameters for ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for processing
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)  # Detect ArUco markers

        # Check if ArUco markers are detected
        if ids is not None and len(corners) > 0:
            print(f"Aruco marker detected with ID: {ids.flatten()}")  # Log detected marker ID
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)  # Draw detected markers on the frame
            marker_corners = corners[0][0]  # Extract the corners of the first detected marker
            side_length_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])  # Calculate side length in pixels
            self.pixel_to_mm = side_length_pixels / aruco_length_mm  # Compute the pixel-to-mm scale factor
            print(f"Scale Factor (Automatic): {self.pixel_to_mm:.3f} pixels/mm")
        else:
            print("No Aruco markers detected. Scale not calculated.")
            self.pixel_to_mm = None

        cv2.imshow("Aruco Marker Detection", frame)  # Display the frame with detected ArUco markers

    # Purpose: Detect objects (e.g., needles) in the frame
    # This method applies edge detection, morphological operations, and contour detection
    # to identify objects in the input frame.
    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        cv2.imshow("Grayscale Image", gray)  # Display the grayscale image

        edges = cv2.Canny(gray, 20, 80)  # Apply Canny edge detection
        cv2.imshow("Canny Edges (Raw)", edges)  # Display raw edges

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Define a rectangular structuring element
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)  # Apply morphological closing
        cv2.imshow("Closed Edges (Morphology)", closed_edges)  # Display processed edges

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        print(f"Number of contours detected: {len(contours)}")  # Log the number of contours detected

        debug_frame = frame.copy()  # Create a copy of the frame for drawing contours
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 2)  # Draw detected contours on the frame
        cv2.imshow("Detected Contours", debug_frame)  # Display the frame with contours

        return contours, closed_edges, gray  # Return detected contours and processed images

    def classify_size(self, length_mm, width_mm):
        # Purpose: Classify an object based on its length and width in millimeters.
        # The classification includes "Size 1 (Small)", "Size 2 (Large)", and "Defect".
        # If the object does not match any predefined category, it returns None.

        # Define size ranges for Size 1 (Small)
        size_1_length_min = 15.0  # Minimum length for Size 1
        size_1_length_max = 22.0  # Maximum length for Size 1
        size_1_width_min = 1.0  # Minimum width for Size 1
        size_1_width_max = 5.5  # Maximum width for Size 1

        # Define size ranges for Size 2 (Large)
        size_2_length_min = 21.0  # Minimum length for Size 2
        size_2_length_max = 27.5  # Maximum length for Size 2
        size_2_width_min = 7.0  # Minimum width for Size 2
        size_2_width_max = 13.0  # Maximum width for Size 2

        # Define defect range
        defect_length_min = 5.0  # Minimum length for defect
        defect_length_max = 18.0  # Maximum length for defect
        defect_width_min = 1.0  # Minimum width for defect
        defect_width_max = 15.0  # Maximum width for defect

        # Check for Size 1 (Small)
        if (size_1_length_min <= length_mm <= size_1_length_max and
                size_1_width_min <= width_mm <= size_1_width_max):
            return "Size 1 (Small)"  # Return classification for Size 1

        # Check for Size 2 (Large)
        elif (size_2_length_min <= length_mm <= size_2_length_max and
              size_2_width_min <= width_mm <= size_2_width_max):
            return "Size 2 (Large)"  # Return classification for Size 2

        # Check for Defect
        elif (defect_length_min <= length_mm <= defect_length_max and
              defect_width_min <= width_mm <= defect_width_max):
            return "Defect"  # Return classification for Defect

        # Default to None if no match
        return None  # Return None if the object does not fall into any category

    def detect_color(self, frame, contour):
        # Purpose: Detect the dominant color within a given contour in an image.
        # The function identifies colors such as Red, Green, Blue, Yellow, White, and Black
        # using HSV color space and returns the most dominant color or "None" if no match is found.

        # Extract the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)  # Get the rectangle enclosing the contour
        roi = frame[y:y + h, x:x + w]  # Region of interest (ROI) containing the object

        # Convert the ROI from BGR to HSV color space for better color segmentation
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges for common colors
        color_ranges = {
            'Red': ((0, 120, 70), (10, 255, 255)),  # HSV range for Red
            'Green': ((35, 50, 50), (85, 255, 255)),  # HSV range for Green
            'Blue': ((100, 50, 50), (140, 255, 255)),  # HSV range for Blue
            'Yellow': ((20, 100, 100), (30, 255, 255)),  # HSV range for Yellow
            'White': ((0, 0, 180), (180, 60, 255)),  # HSV range for White
            'Black': ((0, 0, 0), (180, 255, 50))  # HSV range for Black
        }

        detected_colors = []  # List to store detected colors and their intensity

        # Iterate over each color range to check for a match in the ROI
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_roi, lower, upper)  # Create a mask for the color range
            if np.sum(mask) > 0:  # Check if the color is present in the ROI
                detected_colors.append((color, np.sum(mask)))  # Add the color and its intensity

        # If any colors are detected, return the most dominant one
        if detected_colors:
            detected_colors.sort(key=lambda x: x[1], reverse=True)  # Sort by intensity in descending order
            return [detected_colors[0][0]]  # Return the color with the highest intensity

        # Return "None" if no colors are detected
        return ["None"]

    def analyze_needles(self, frame, gray, contours):
        # Purpose: Analyze needle-like objects in an image frame.
        # The function processes contours, classifies the size, detects color, checks for crossed tips,
        # and annotates the frame with relevant information.

        result_frame = frame.copy()  # Create a copy of the frame for annotations

        # Iterate through each contour to analyze needle properties
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)  # Calculate contour area

            # Skip contours with area outside the valid range
            if area < 100 or area > 2500:
                continue

            # Get the minimum area bounding rectangle for the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # Get corner points of the rectangle
            box = box.astype(np.int32)

            # Extract rectangle properties: center, dimensions, and angle
            (x, y), (w, h), angle = rect
            length, width = max(w, h), min(w, h)  # Determine the longer and shorter sides

            # Convert dimensions from pixels to millimeters if scale is available
            if self.pixel_to_mm:
                length_mm = length / self.pixel_to_mm
                width_mm = width / self.pixel_to_mm
            else:
                length_mm, width_mm = length, width  # Use pixel values if scale is unavailable

            # Classify the needle size based on dimensions
            size_label = self.classify_size(length_mm, width_mm)

            # Skip the contour if its size is not recognized
            if size_label is None:
                continue

            # Detect the dominant color(s) of the needle
            detected_colors = self.detect_color(frame, contour)
            color_text = ", ".join(detected_colors) if detected_colors else "None"

            # Set the border color based on size classification
            if size_label == "Defect":
                border_color = (0, 0, 255)  # Red for defects
            else:
                border_color = (0, 255, 0)  # Green for valid sizes

            # Draw the bounding box around the needle
            cv2.polylines(result_frame, [box], True, border_color, 2)

            # Check if the needle's tip is crossed with another contour
            crossed_tips = False
            for other_contour in contours:
                if np.array_equal(contour, other_contour):  # Skip the same contour
                    continue
                # Calculate intersection area between contours
                intersection_area = cv2.intersectConvexConvex(np.float32(contour), np.float32(other_contour))[0]
                if intersection_area > 0:  # If contours overlap
                    crossed_tips = True
                    break

            # Create text annotations with needle details
            text_lines = [
                f"Size: {size_label}",
                f"L: {length_mm:.1f} mm, W: {width_mm:.1f} mm",
                f"Angle: {angle:.1f} degrees",
                f"Colors: {color_text}",
                f"Crossed Tips: {'Yes' if crossed_tips else 'No'}"
            ]

            # Determine text placement near the needle
            box_center = np.mean(box, axis=0)  # Calculate the center of the bounding box
            text_x, text_y = int(box_center[0] + 20), int(box_center[1])  # Offset for text placement
            text_box_width = 180
            text_box_height = 20 * len(text_lines) + 10
            padding = 5

            # Draw a semi-transparent background for the text
            overlay = result_frame.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y),
                (text_x + text_box_width, text_y + text_box_height),
                (0, 0, 0),  # Black background
                -1
            )
            alpha = 0.6  # Set transparency level
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)

            # Draw each line of text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            line_spacing = 20
            current_y = text_y + padding + 15

            for line in text_lines:
                cv2.putText(
                    result_frame,
                    line,
                    (text_x + padding, current_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness,
                    lineType=cv2.LINE_AA
                )
                current_y += line_spacing  # Move to the next line

            # Log the needle details in the console
            print(
                f"Needle {idx + 1}: {size_label} | Length: {length_mm:.1f}mm | Width: {width_mm:.1f}mm | Angle: {angle:.1f} | Colors: {color_text} | Crossed Tips: {'Yes' if crossed_tips else 'No'}"
            )

        # Return the annotated frame
        return result_frame

if __name__ == "__main__":
    # Purpose: Main entry point for the needle detection and analysis program.
    # This script initializes the camera, processes live video feed, and performs
    # needle detection, size classification, color detection, and visualization.

    cap = cv2.VideoCapture(1)  # Open camera at index 1 (adjust if necessary)

    # Check if the camera is accessible
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    # Set camera resolution to 1280x720 for better visualization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = NeedleDetector()  # Initialize the needle detection class

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")  # Exit if frame is not captured
            break

        # Step 1: Detect ArUco markers in the frame to calculate scale
        detector.calculate_scale_from_aruco(frame)

        # Step 2: Detect objects (contours) and analyze needles
        contours, _, gray = detector.detect_objects(frame)  # Detect edges and contours
        result_frame = detector.analyze_needles(frame, gray, contours)  # Analyze detected needles

        # Step 3: Display the result with annotations
        cv2.imshow("Needle Analysis", result_frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera resources and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

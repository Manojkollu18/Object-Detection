# YOLOv8 Object Detection Project
# ================================
# Features: Webcam input, file upload, batch processing of multiple images

# Cell 1: Import Required Libraries and Setup
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import glob
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, Image, clear_output
import base64
from io import BytesIO
from PIL import Image as PILImage
import warnings
warnings.filterwarnings('ignore')

#Cell 2: Install and Import YOLOv8
print("Libraries imported successfully!")
# Run this cell first to install ultralytics if not already installed
#!pip install ultralytics

from ultralytics import YOLO
print("YOLOv8 imported successfully!")
# Cell 3: Load Pretrained YOLOv8 Model
class ObjectDetector:
    """
    Object Detection class using YOLOv8 pretrained model
    """
    def __init__(self, model_name='yolov8l.pt'):
        """
        Initialize the object detector

        Args:
            model_name (str): Name of the YOLOv8 model to use
        """
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        print(f"{model_name} model loaded successfully!")

        # Get class names from the model
        self.class_names = self.model.names
        print(f"Model can detect {len(self.class_names)} different classes")

    def detect_objects(self, image_path, confidence_threshold=0.5):
        """
        Detect objects in a single image

        Args:
            image_path (str): Path to the image file
            confidence_threshold (float): Minimum confidence for detection

        Returns:
            dict: Detection results with image and annotations
        """
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)

        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detection_data = {
            'image': image_rgb,
            'detections': [],
            'image_path': image_path
        }

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]

                    detection_data['detections'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })

        return detection_data

    def detect_multiple_images(self, image_paths, confidence_threshold=0.5):
        """
        Detect objects in multiple images

        Args:
            image_paths (list): List of image file paths
            confidence_threshold (float): Minimum confidence for detection

        Returns:
            list: List of detection results for each image
        """
        results_list = []

        print(f"Processing {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.detect_objects(image_path, confidence_threshold)
            results_list.append(result)

        print("Batch processing completed!")
        return results_list

# Initialize the detector
detector = ObjectDetector('yolov8l.pt')
# Cell 4: Visualization Functions
class DetectionVisualizer:
    """
    Class for visualizing object detection results
    """
    def __init__(self):
        # Define colors for different classes (you can customize these)
        self.colors = plt.cm.Set3(np.linspace(0, 1, 100))

    def draw_detections(self, detection_data, figsize=(12, 8)):
        """
        Draw bounding boxes on image

        Args:
            detection_data (dict): Detection results from ObjectDetector
            figsize (tuple): Figure size for matplotlib
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Display image
        ax.imshow(detection_data['image'])
        ax.set_title(f"Detections: {os.path.basename(detection_data['image_path'])}")
        ax.axis('off')

        # Draw bounding boxes
        for detection in detection_data['detections']:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1

            # Get color for this class
            color = self.colors[detection['class_id'] % len(self.colors)]

            # Draw rectangle
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor=color,
                           facecolor='none', alpha=0.8)
            ax.add_patch(rect)

            # Add label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            ax.text(x1, y1-5, label, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   color='black', weight='bold')

        plt.tight_layout()
        plt.show()

        # Print detection summary
        print(f"\nDetection Summary for {os.path.basename(detection_data['image_path'])}:")
        print(f"Total objects detected: {len(detection_data['detections'])}")

        # Count objects by class
        class_counts = {}
        for detection in detection_data['detections']:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name, count in class_counts.items():
            print(f"- {class_name}: {count}")

    def display_multiple_results(self, results_list, max_cols=3):
        """
        Display results for multiple images in a grid

        Args:
            results_list (list): List of detection results
            max_cols (int): Maximum columns in the grid
        """
        n_images = len(results_list)
        n_cols = min(max_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

        # Ensure axes is always an iterable of Axes objects
        axes = axes.flatten()

        for i, detection_data in enumerate(results_list):
            if i < len(axes):
                ax = axes[i]

                # Display image
                ax.imshow(detection_data['image'])
                ax.set_title(f"{os.path.basename(detection_data['image_path'])}\n"
                           f"Objects: {len(detection_data['detections'])}")
                ax.axis('off')

                # Draw bounding boxes
                for detection in detection_data['detections']:
                    x1, y1, x2, y2 = detection['bbox']
                    width = x2 - x1
                    height = y2 - y1

                    # Get color for this class
                    color = self.colors[detection['class_id'] % len(self.colors)]

                    # Draw rectangle
                    rect = Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor=color,
                                   facecolor='none', alpha=0.8)
                    ax.add_patch(rect)

        # Hide unused subplots
        for i in range(len(results_list), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        # Print overall summary
        total_objects = sum(len(result['detections']) for result in results_list)
        print(f"\nBatch Processing Summary:")
        print(f"Total images processed: {len(results_list)}")
        print(f"Total objects detected: {total_objects}")

# Initialize visualizer
visualizer = DetectionVisualizer()
# Cell 5: Webcam Capture Function
def capture_from_webcam(save_path="webcam_capture.jpg"):
    """
    Capture image from webcam

    Args:
        save_path (str): Path to save the captured image

    Returns:
        str: Path to the saved image or None if capture failed
    """
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None

    print("Webcam initialized. Press SPACE to capture, ESC to exit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        # Display frame
        cv2.imshow('Webcam - Press SPACE to capture, ESC to exit', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Space key
            cv2.imwrite(save_path, frame)
            print(f"Image captured and saved as {save_path}")
            break
        elif key == 27:  # Escape key
            print("Capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return save_path
# Cell 6: File Upload and Selection Functions
def select_single_image():
    """
    Interactive file selection for a single image
    """
    print("Please enter the path to your image file:")
    print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")

    file_path = input("Image path: ").strip().strip('"').strip("'")

    if os.path.exists(file_path):
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            return file_path
        else:
            print("Error: Unsupported file format")
            return None
    else:
        print("Error: File not found")
        return None

def select_multiple_images():
    """
    Interactive file selection for multiple images
    """
    print("Select multiple images:")
    print("Option 1: Enter a directory path to process all images in that directory")
    print("Option 2: Enter multiple file paths separated by semicolon (;)")

    choice = input("Enter '1' for directory or '2' for individual files: ").strip()

    if choice == '1':
        dir_path = input("Directory path: ").strip().strip('"').strip("'")

        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Get all image files in directory
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []

            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(dir_path, ext)))

            if image_files:
                print(f"Found {len(image_files)} images in directory")
                return image_files
            else:
                print("No image files found in directory")
                return []
        else:
            print("Error: Directory not found")
            return []

    elif choice == '2':
        file_paths_str = input("Enter file paths separated by semicolon (;): ").strip()
        file_paths = [path.strip().strip('"').strip("'") for path in file_paths_str.split(';')]

        valid_paths = []
        for path in file_paths:
            if os.path.exists(path) and path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                valid_paths.append(path)
            else:
                print(f"Warning: Skipping invalid file: {path}")

        return valid_paths

    else:
        print("Invalid choice")
        return []
# Cell 7: Main Interactive Interface
def run_object_detection():
    """
    Main function to run object detection with user interface
    """
    print("=" * 60)
    print("YOLOv8 Object Detection System")
    print("=" * 60)

    while True:
        print("\nSelect input method:")
        print("1. Webcam capture")
        print("2. Single image file")
        print("3. Multiple image files")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            print("\n--- Webcam Capture ---")
            image_path = capture_from_webcam()

            if image_path:
                # Set confidence threshold
                conf = float(input("Enter confidence threshold (0.0-1.0, default 0.5): ") or "0.5")

                # Detect objects
                result = detector.detect_objects(image_path, confidence_threshold=conf)

                # Visualize results
                visualizer.draw_detections(result)

        elif choice == '2':
            print("\n--- Single Image Detection ---")
            image_path = select_single_image()

            if image_path:
                # Set confidence threshold
                conf = float(input("Enter confidence threshold (0.0-1.0, default 0.5): ") or "0.5")

                # Detect objects
                result = detector.detect_objects(image_path, confidence_threshold=conf)

                # Visualize results
                visualizer.draw_detections(result)

        elif choice == '3':
            print("\n--- Multiple Images Detection ---")
            image_paths = select_multiple_images()

            if image_paths:
                # Set confidence threshold
                conf = float(input("Enter confidence threshold (0.0-1.0, default 0.5): ") or "0.5")

                # Detect objects in all images
                results = detector.detect_multiple_images(image_paths, confidence_threshold=conf)

                # Visualize results
                visualizer.display_multiple_results(results)

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

# Display available classes
print("\nYOLOv8 can detect the following classes:")
print("-" * 40)
for i, class_name in enumerate(detector.class_names.values()):
    print(f"{i}: {class_name}")
print(f"\nTotal classes: {len(detector.class_names)}")

# Cell 8: Run the Application

print("\nSetup completed! Run the next cell to start object detection.")

# Cell 9: Start Object Detection

run_object_detection()

print("Ready to start object detection!")
print("Run: run_object_detection() to begin")

#cell 10: Example usage
#def example_usage():
    #"""
    #Example of direct usage without interactive interface
    #"""
    #print("Example: Direct usage of object detection functions")

    # Example 1: Process a single image (replace with your image path)
    #image_path = "image1.jpg"
    #result = detector.detect_objects(image_path, confidence_threshold=0.5)
    #visualizer.draw_detections(result)

    # Example 2: Process multiple images
    #image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    #results = detector.detect_multiple_images(image_paths, confidence_threshold=0.5)
    #visualizer.display_multiple_results(results)

    #print("Replace the commented code above with your actual image paths to test")

#example_usage()    


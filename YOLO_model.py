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
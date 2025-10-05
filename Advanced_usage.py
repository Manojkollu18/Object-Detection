# Cell 11: Advanced Features and Customization
class AdvancedDetector(ObjectDetector):
    """
    Extended detector with additional features
    """
    def __init__(self, model_name='yolov8l.pt'):
        super().__init__(model_name)

    def detect_with_filtering(self, image_path, target_classes=None, confidence_threshold=0.5, size_threshold=0.01):
        """
        Detect objects with class and size filtering

        Args:
            image_path (str): Path to image
            target_classes (list): List of class names to detect (None for all)
            confidence_threshold (float): Minimum confidence
            size_threshold (float): Minimum relative size (bbox area / image area)
        """
        result = self.detect_objects(image_path, confidence_threshold)

        if target_classes is not None:
            # Filter by target classes
            filtered_detections = []
            for detection in result['detections']:
                if detection['class_name'] in target_classes:
                    filtered_detections.append(detection)
            result['detections'] = filtered_detections

        # Filter by size
        if size_threshold > 0:
            image_height, image_width = result['image'].shape[:2]
            image_area = image_height * image_width

            size_filtered_detections = []
            for detection in result['detections']:
                x1, y1, x2, y2 = detection['bbox']
                bbox_area = (x2 - x1) * (y2 - y1)
                relative_size = bbox_area / image_area

                if relative_size >= size_threshold:
                    size_filtered_detections.append(detection)

            result['detections'] = size_filtered_detections

        return result

    def get_detection_statistics(self, results_list):
        """
        Get detailed statistics from detection results

        Args:
            results_list (list): List of detection results

        Returns:
            dict: Statistics summary
        """
        stats = {
            'total_images': len(results_list),
            'total_objects': 0,
            'class_counts': {},
            'confidence_stats': [],
            'images_with_detections': 0
        }

        for result in results_list:
            if result['detections']:
                stats['images_with_detections'] += 1

            for detection in result['detections']:
                stats['total_objects'] += 1
                class_name = detection['class_name']
                stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
                stats['confidence_stats'].append(detection['confidence'])

        if stats['confidence_stats']:
            stats['avg_confidence'] = np.mean(stats['confidence_stats'])
            stats['min_confidence'] = np.min(stats['confidence_stats'])
            stats['max_confidence'] = np.max(stats['confidence_stats'])

        return stats

# Initialize advanced detector
advanced_detector = AdvancedDetector('yolov8l.pt')
print("Advanced features ready!")
print("Uncomment the advanced_detector initialization to use advanced features")


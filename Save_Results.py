# Cell 12: Save and Export Results
def save_results_to_file(results_list, output_file="detection_results.txt"):
    """
    Save detection results to a text file

    Args:
        results_list (list): List of detection results
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write("YOLOv8 Object Detection Results\n")
        f.write("=" * 40 + "\n\n")

        for i, result in enumerate(results_list):
            f.write(f"Image {i+1}: {os.path.basename(result['image_path'])}\n")
            f.write(f"Objects detected: {len(result['detections'])}\n")

            for j, detection in enumerate(result['detections']):
                f.write(f"  Object {j+1}:\n")
                f.write(f"    Class: {detection['class_name']}\n")
                f.write(f"    Confidence: {detection['confidence']:.3f}\n")
                f.write(f"    Bounding Box: {detection['bbox']}\n")

            f.write("\n")

    print(f"Results saved to {output_file}")

print("Object Detection Project Setup Complete!")
print("=" * 60)
print("Available functions:")
print("- run_object_detection(): Start interactive detection system")
print("- capture_from_webcam(): Capture image from webcam")
print("- detector.detect_objects(): Detect objects in single image")
print("- detector.detect_multiple_images(): Detect objects in multiple images")
print("- visualizer.draw_detections(): Visualize detection results")
print("- save_results_to_file(): Save results to text file")
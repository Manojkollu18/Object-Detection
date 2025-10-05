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


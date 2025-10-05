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
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
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
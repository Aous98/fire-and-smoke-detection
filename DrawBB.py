import os
import cv2
import numpy as np
from pathlib import Path


def visualize_yolo_annotations(images_dir, labels_dir, output_dir, class_names):
    """
    Draw YOLO format bounding boxes on images and save the results.

    Args:
        images_dir (str): Directory containing original images
        labels_dir (str): Directory containing YOLO TXT files
        output_dir (str): Directory to save visualized images
        class_names (dict): Mapping of class IDs to names (e.g., {0: 'smoke', 1: 'fire'})
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Colors for different classes (BGR format)
    colors = {
        0: (0, 255, 0),  # Green for smoke
        1: (0, 0, 255)  # Red for fire
    }

    for image_file in image_files:
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue

        img_height, img_width = image.shape[:2]

        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"No labels found for {image_file}")
            # Save the original image if no annotations
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, image)
            continue

        # Read YOLO format annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Process each annotation
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO format to pixel coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # Calculate bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width - 1, x2)
            y2 = min(img_height - 1, y2)

            # Draw bounding box
            color = colors.get(class_id, (0, 255, 0))  # Default to green if unknown class
            thickness = max(2, int(min(img_width, img_height) / 300))  # Dynamic thickness
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Add class label
            label = class_names.get(class_id, f"class_{class_id}")
            font_scale = min(img_width, img_height) / 1000
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save the visualized image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)
        print(f"Processed: {image_file}")


# Example usage
if __name__ == "__main__":
    # Configuration - update these paths to match your setup
    IMAGES_DIR = "/home/aous/Desktop/personal projects/FireAndSmoke/data/train"  # Directory with original images
    LABELS_DIR = "/home/aous/Desktop/personal projects/FireAndSmoke/data/train"  # Directory with YOLO TXT files
    OUTPUT_DIR = "/home/aous/Desktop/personal projects/FireAndSmoke/data/imageswithbb/t"  # Where to save images with boxes

    # Class names mapping (must match your YOLO class indices)
    CLASS_NAMES = {
        0: "smoke",
        1: "fire"
    }

    visualize_yolo_annotations(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES
    )

    print(f"\nVisualization complete! Results saved to: {OUTPUT_DIR}")
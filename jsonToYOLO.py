import json
import os
from pathlib import Path


def convert_coco_to_yolo(json_path, output_dir, classes_of_interest=None):
    """
    Convert COCO JSON annotations to YOLO format TXT files, filtering for specific classes.

    Args:
        json_path (str): Path to COCO JSON file
        output_dir (str): Directory to save YOLO TXT files
        classes_of_interest (list): List of class IDs to include (None for all classes)
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create a mapping from category ID to YOLO class index
    # We'll only include fire and smoke in our mapping
    category_map = {}
    yolo_class_index = 0

    for category in data['categories']:
        if classes_of_interest is None or category['id'] in classes_of_interest:
            if category['name'] == 'fire' or category['name'] == 'smoke':
                category_map[category['id']] = yolo_class_index
                yolo_class_index += 1

    # Print the category mapping for verification
    print("Category to YOLO class mapping:")
    for cat_id, yolo_id in category_map.items():
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        print(f"  COCO {cat_name} (id:{cat_id}) -> YOLO class {yolo_id}")

    # Create a dictionary to map image IDs to file names
    image_id_to_info = {img['id']: img for img in data['images']}

    # Process all annotations
    for annotation in data['annotations']:
        # Only process annotations for fire and smoke
        if annotation['category_id'] not in category_map:
            continue

        # Get corresponding image info
        image_info = image_id_to_info.get(annotation['image_id'])
        if not image_info:
            continue  # Skip if image not found

        # Prepare YOLO format line
        x, y, w, h = annotation['bbox']
        image_width, image_height = image_info['width'], image_info['height']

        # Calculate normalized coordinates
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        width_norm = w / image_width
        height_norm = h / image_height

        # Get YOLO class ID
        yolo_class = category_map[annotation['category_id']]

        # Prepare the line to write to file
        yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"

        # Determine output TXT file path
        image_filename = os.path.basename(image_info['file_name'])
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        # Append the annotation to the file
        with open(txt_path, 'a') as f:
            f.write(yolo_line)

    # Create empty TXT files for images with no fire/smoke annotations
    all_image_ids = set(image_id_to_info.keys())
    annotated_image_ids = {ann['image_id'] for ann in data['annotations']
                           if ann['category_id'] in category_map}

    for image_id in all_image_ids - annotated_image_ids:
        image_info = image_id_to_info[image_id]
        image_filename = os.path.basename(image_info['file_name'])
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        # Create empty file
        with open(txt_path, 'w') as f:
            pass

    print(f"\nConversion complete! TXT files saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Specify your paths here
    coco_json_path = "/home/aous/Desktop/personal projects/FireAndSmoke/dataset_fire_smoke/train/train.json"  # Path to your COCO JSON file
    output_directory = "/home/aous/Desktop/personal projects/FireAndSmoke/dataset_fire_smoke/train/annotations"
    # Only include fire (id:5) and smoke (id:4) from your categories
    classes_to_include = [4, 5]  # These are the category_ids from your JSON

    convert_coco_to_yolo(
        json_path=coco_json_path,
        output_dir=output_directory,
        classes_of_interest=classes_to_include
    )

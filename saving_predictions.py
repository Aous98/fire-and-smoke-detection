import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

def visualize_validation_results(model_path, val_dir, output_dir, conf_thres=0.3):
    """
    Save validation images with both predicted and ground truth boxes
    
    Args:
        model_path (str): Path to YOLOv11 model weights (.pt file)
        val_dir (str): Path to validation images directory
        output_dir (str): Directory to save visualized images
        conf_thres (float): Confidence threshold
    """
    # Verify model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded model from {model_path}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all validation images
    val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Colors (BGR format)
    pred_color = (0, 255, 0)    # Green for predictions
    gt_color = (0, 0, 255)      # Red for ground truth
    text_color = (255, 255, 255) # White text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for img_name in tqdm(val_images, desc="Processing validation images"):
        img_path = os.path.join(val_dir, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not load image {img_path}")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Get corresponding label file
        label_path = os.path.join(
            val_dir.replace('images', 'labels'), 
            img_name.rsplit('.', 1)[0] + '.txt'
        )
        
        # Draw ground truth boxes if label exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        
                        # Convert YOLO format to pixel coordinates
                        x_center *= img_w
                        y_center *= img_h
                        width *= img_w
                        height *= img_h
                        x1 = max(0, int(x_center - width/2))
                        y1 = max(0, int(y_center - height/2))
                        x2 = min(img_w, int(x_center + width/2))
                        y2 = min(img_h, int(y_center + height/2))
                        
                        # Draw ground truth box
                        cv2.rectangle(img, (x1, y1), (x2, y2), gt_color, 2)
                        cv2.putText(img, f"GT:{['smoke','fire'][int(class_id)]}", 
                                    (x1, y1-10), font, 0.5, text_color, 1)
                    except Exception as e:
                        print(f"⚠️ Error processing label {label_path}: {e}")
        
        # Get model predictions
        try:
            results = model.predict(img_path, conf=conf_thres)
            
            # Draw predicted boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                class_id = int(box.cls.item())
                
                # Ensure boxes are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                
                # Draw prediction box
                cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, 2)
                cv2.putText(img, f"Pred:{['smoke','fire'][class_id]} {conf:.2f}", 
                            (x1, y1-30), font, 0.5, text_color, 1)
        except Exception as e:
            print(f"⚠️ Prediction failed for {img_path}: {e}")
        
        # Add legend
        cv2.rectangle(img, (5, 5), (300, 80), (0,0,0), -1)  # Background
        cv2.putText(img, "Ground Truth (Red)", (10, 30), font, 0.7, gt_color, 2)
        cv2.putText(img, "Predictions (Green)", (10, 60), font, 0.7, pred_color, 2)
        
        # Save image
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)
        print(f"💾 Saved visualization: {output_path}") if len(val_images) <= 5 else None

    print(f"\n✅ Completed! Saved {len(val_images)} images to {output_dir}")

# Example Usage
if __name__ == "__main__":
    visualize_validation_results(
        model_path="/home/aous/Desktop/personal projects/fire and smoke detection/runs/detect/train5/weights/best.pt",  # Path to your trained model
        val_dir="/home/aous/Desktop/personal projects/FireAndSmoke/data/val",  # Validation images
        output_dir="/home/aous/Desktop/personal projects/FireAndSmoke/data/result",  # Output directory
        conf_thres=0.3  # Confidence threshold
    )
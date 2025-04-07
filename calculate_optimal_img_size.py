import json
import numpy as np
from pathlib import Path

def calculate_optimal_size(json_path):
    """
    Calculate optimal YOLO training size from JSON annotations.
    Returns: Recommended size (multiple of 32), median, and mean dimensions.
    """
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)
    
    # Extract image dimensions
    widths = []
    heights = []
    
    for img in data['images']:
        widths.append(img['width'])
        heights.append(img['height'])
    
    # Calculate statistics
    median_width = np.median(widths)
    median_height = np.median(heights)
    mean_width = np.mean(widths)
    mean_height = np.mean(heights)
    
    # Recommended YOLO size must be multiple of 32
    def round_to_multiple(value, multiple=32):
        return multiple * round(value / multiple)
    
    # Calculate recommendations
    median_rec = round_to_multiple(np.median([median_width, median_height]))
    mean_rec = round_to_multiple(np.mean([mean_width, mean_height]))
    
    # Choose the most common standard size
    standard_sizes = [320, 416, 512, 640, 768, 896, 1024, 1280]
    recommended_size = min(standard_sizes, key=lambda x: abs(x - median_rec))
    
    return {
        'recommended_size': recommended_size,
        'median_dimensions': (median_width, median_height),
        'mean_dimensions': (mean_width, mean_height),
        'size_options': standard_sizes,
        'all_widths': widths,
        'all_heights': heights
    }

# Usage
if __name__ == "__main__":
    json_path = "/home/aous/Desktop/personal projects/FireAndSmoke/data/train.json"  # Path to your COCO JSON file
    stats = calculate_optimal_size(json_path)
    
    print("\nImage Size Analysis:")
    print(f"- Number of images: {len(stats['all_widths'])}")
    print(f"- Median dimensions: {stats['median_dimensions'][0]}x{stats['median_dimensions'][1]}")
    print(f"- Mean dimensions: {stats['mean_dimensions'][0]:.1f}x{stats['mean_dimensions'][1]:.1f}")
    print(f"\nRecommended YOLO training size: {stats['recommended_size']}x{stats['recommended_size']}")
    print(f"Standard size options: {stats['size_options']}")
    
    # Visualization
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Width vs Height scatter
        plt.subplot(1, 2, 1)
        plt.scatter(stats['all_widths'], stats['all_heights'], alpha=0.6)
        plt.title("Image Dimensions Distribution")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.grid(True)
        
        # Size histogram
        plt.subplot(1, 2, 2)
        areas = [w*h for w,h in zip(stats['all_widths'], stats['all_heights'])]
        plt.hist(areas, bins=30, edgecolor='black')
        plt.title("Image Area Distribution")
        plt.xlabel("Total Pixels (width × height)")
        plt.ylabel("Count")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("image_size_analysis.png", dpi=120)
        print("\nVisualization saved to 'image_size_analysis.png'")
    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")

    print("\nTraining Recommendations:")
    if stats['recommended_size'] <= 640:
        print("- Use batch=8-16 for good speed/accuracy balance")
    else:
        print("- Use batch=4-8 (reduce batch size for larger images)")
    print("- Enable mosaic augmentation for better small object detection")
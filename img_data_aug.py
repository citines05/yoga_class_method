from PIL import Image, ImageOps
import os
import numpy as np

# Main directory of dataset
dataset_dir = 'dataset'  # Replace with the actual dataset path
train_dir = os.path.join(dataset_dir, 'train')  # Work only with the training set

# Iterate over classes in the training directory
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip non-directory files

    # Count existing images in the class
    current_images = os.listdir(class_dir)
    current_image_count = len(current_images)

    print(f"Performing horizontal flip for images in class '{class_name}'...")

    # Perform horizontal flip on each image in the class
    for image_name in current_images:
        image_path = os.path.join(class_dir, image_name)
        try:
            img = Image.open(image_path).convert('RGB')  # Convert to RGB
            img_flipped = ImageOps.mirror(img)  # Apply horizontal flip
            flipped_image_name = f"flipped_{os.path.splitext(image_name)[0]}.jpg"
            img_flipped.save(os.path.join(class_dir, flipped_image_name))  # Save the flipped image
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

    print(f"Completed for class '{class_name}'. Total images after flip: {len(os.listdir(class_dir))}")

print("Horizontal flip completed for all classes!")

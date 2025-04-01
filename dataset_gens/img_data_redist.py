import os
import shutil
from math import floor, ceil

# Dataset 1 directories
dataset_dir = 'dataset'  # Replace with the actual dataset path
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

# Check if the directories exist
if not os.path.exists(valid_dir) or not os.path.exists(test_dir):
    raise ValueError("The 'valid' and 'test' directories do not exist in the provided dataset.")

# Redistribute images for each class
for class_name in os.listdir(valid_dir):
    valid_class_dir = os.path.join(valid_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    # Skip non-directory files
    if not os.path.isdir(valid_class_dir):
        continue

    # Create the corresponding test directory if it doesn't exist
    if not os.path.exists(test_class_dir):
        os.makedirs(test_class_dir)

    # List all images in validation and test for the class
    valid_images = os.listdir(valid_class_dir)
    test_images = os.listdir(test_class_dir)

    # Total number of images for this class
    total_images = valid_images + test_images

    # Calculate the new 50/50 split
    total_count = len(total_images)
    new_valid_count = floor(total_count / 2)
    new_test_count = ceil(total_count / 2)  # Excess goes to the test set

    print(f"Class '{class_name}': Total images = {total_count}. "
          f"New splits: validation = {new_valid_count}, test = {new_test_count}")

    # Move all images to a temporary directory
    temp_dir = os.path.join(dataset_dir, 'temp', class_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for image in valid_images:
        shutil.move(os.path.join(valid_class_dir, image), temp_dir)
    for image in test_images:
        shutil.move(os.path.join(test_class_dir, image), temp_dir)

    # Reshuffle images to redistribute
    all_images = os.listdir(temp_dir)
    all_images.sort()  # Optional: sort for consistency
    new_valid_images = all_images[:new_valid_count]
    new_test_images = all_images[new_valid_count:]

    # Move back to validation
    for image in new_valid_images:
        shutil.move(os.path.join(temp_dir, image), valid_class_dir)

    # Move back to test
    for image in new_test_images:
        shutil.move(os.path.join(temp_dir, image), test_class_dir)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

print("Redistribution completed!")

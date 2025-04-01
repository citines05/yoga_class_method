import os
import random
import shutil
from tqdm import tqdm

# Base directory
base_dir = "datasets"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "valid")

# Create test and validation directories if they don't exist
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Function to split the data
def split_data():
    for class_name in tqdm(os.listdir(train_dir), desc="Processing classes"):
        class_train_dir = os.path.join(train_dir, class_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(class_train_dir):
            continue
        
        # Create corresponding directories for test and validation
        class_test_dir = os.path.join(test_dir, class_name)
        class_valid_dir = os.path.join(valid_dir, class_name)
        os.makedirs(class_test_dir, exist_ok=True)
        os.makedirs(class_valid_dir, exist_ok=True)
        
        # List all images of the class
        images = os.listdir(class_train_dir)
        random.shuffle(images)
        
        # Calculate splits
        num_images = len(images)
        num_test = num_images * 30 // 100  # 30% for testing
        num_train = num_images - num_test  # 70% for training
        
        # Split data
        train_images = images[:num_train]
        test_images = images[num_train:num_train + num_test]
        
        # Split test into validation
        num_valid = len(test_images) // 2  # Half for validation
        valid_images = test_images[:num_valid]
        test_images = test_images[num_valid:]  # Remaining for testing
        
        # Move files
        for img in train_images:
            shutil.move(
                os.path.join(class_train_dir, img),
                os.path.join(class_train_dir, img)  # Stays in training
            )
        for img in valid_images:
            shutil.move(
                os.path.join(class_train_dir, img),
                os.path.join(class_valid_dir, img)
            )
        for img in test_images:
            shutil.move(
                os.path.join(class_train_dir, img),
                os.path.join(class_test_dir, img)
            )

# Execute the split
split_data()

print("Data splitting completed!")

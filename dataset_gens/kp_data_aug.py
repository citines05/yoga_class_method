import os
import numpy as np
from tqdm import tqdm

# Main directories
dataset_dir = 'kp_ds_aug_balan_3'
train_dir = os.path.join(dataset_dir, 'train')  # Work only with the training set

# Function to apply random perturbation
def add_noise(keypoints, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, size=keypoints[:, :3].shape)  # Gaussian noise
    keypoints[:, :3] += noise
    return keypoints

# Function to combine keypoints
def combine_keypoints(keypoints1, keypoints2, alpha_range=(0.3, 0.7)):
    alpha = np.random.uniform(*alpha_range)
    return alpha * keypoints1 + (1 - alpha) * keypoints2

# Get the class with the highest number of files
class_counts = {
    class_name: len(os.listdir(os.path.join(train_dir, class_name)))
    for class_name in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, class_name))
}
max_files = max(class_counts.values())

# Iterate over classes to perform data augmentation
for class_name, current_count in class_counts.items():
    class_dir = os.path.join(train_dir, class_name)
    keypoint_files = [
        f for f in os.listdir(class_dir) if f.endswith('.npy')
    ]
    
    print(f"Processing class '{class_name}' ({current_count}/{max_files} files)...")

    if current_count >= max_files:
        print(f"Class '{class_name}' is already balanced. No action needed.")
        continue

    files_to_generate = max_files - current_count
    generated_count = 0

    with tqdm(total=files_to_generate, desc=f"Generating for '{class_name}'") as pbar:
        while generated_count < files_to_generate:
            # Select two random keypoints to combine
            keypoints1 = np.load(os.path.join(class_dir, np.random.choice(keypoint_files)))
            keypoints2 = np.load(os.path.join(class_dir, np.random.choice(keypoint_files)))

            # Apply data augmentation
            if np.random.rand() > 0.5:  # 50% chance to choose the technique
                augmented_keypoints = add_noise(keypoints1)
            else:
                augmented_keypoints = combine_keypoints(keypoints1, keypoints2)

            # Save augmented keypoints
            save_path = os.path.join(class_dir, f"aug_{class_name}_{generated_count + current_count}.npy")
            np.save(save_path, augmented_keypoints)
            generated_count += 1
            pbar.update(1)  # Update progress bar

    print(f"Completed for class '{class_name}'. Total files generated: {generated_count}")

print("Data augmentation completed for the entire training set!")

import os
import cv2
import numpy as np
import mediapipe as mp
import glob
import tensorflow as tf
from absl import logging

# Configure TensorFlow to suppress informational messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Original dataset directory and new dataset directory for keypoints
root_dir = "dataset" # Root directory name
output_dir = "kp_ds_aug_4"  # Final directory name, change to desired final name

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to extract keypoints from an image using MediaPipe and center them
def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_world_landmarks:
        keypoints = np.zeros((33, 4))
        for i, landmark in enumerate(results.pose_world_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        return keypoints
    return np.zeros((33, 4))  # Return empty keypoints if none are detected

# Function to create the directory structure
def create_directory_structure(output_dir, root_dir):
    os.makedirs(output_dir, exist_ok=True)
    for set_name in ['train', 'test', 'valid']:
        set_dir = os.path.join(root_dir, set_name)
        if not os.path.exists(set_dir):
            print(f"Directory {set_dir} not found. Skipping...")
            continue

        output_set_dir = os.path.join(output_dir, set_name)
        os.makedirs(output_set_dir, exist_ok=True)

        classes = os.listdir(set_dir)
        for class_name in classes:
            class_dir = os.path.join(set_dir, class_name)
            if os.path.isdir(class_dir):  # Confirm it's a class directory
                output_class_dir = os.path.join(output_set_dir, class_name)
                os.makedirs(output_class_dir, exist_ok=True)

# Function to process images and save keypoints
def process_and_save_keypoints(root_dir, output_dir):
    for set_name in ['train', 'test', 'valid']:
        print(f"\n Accessing directory {set_name} \n")
        set_dir = os.path.join(root_dir, set_name)
        if not os.path.exists(set_dir):
            continue

        classes = os.listdir(set_dir)
        for class_name in classes:
            class_dir = os.path.join(set_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            output_class_dir = os.path.join(output_dir, set_name, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            image_paths = glob.glob(os.path.join(class_dir, "*.*"))
            print(f"--- Processing {len(image_paths)} images in class '{class_name}'...")

            for img_path in image_paths:
                image_name = os.path.basename(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"*** Error loading image. Deleting {img_path}")
                    os.remove(img_path)
                    continue

                keypoints = extract_keypoints(image)
                if keypoints is not None:
                    # Save keypoints as a .npy file
                    keypoints_file = os.path.join(output_class_dir, image_name.replace('.jpg', '.npy').replace('.png', '.npy'))
                    np.save(keypoints_file, keypoints)
                else:
                    print(f">>> No keypoints detected in image: {img_path}")

# Create the directory structure and process the keypoints
create_directory_structure(output_dir, root_dir)
process_and_save_keypoints(root_dir, output_dir)

print(f"Processing completed. Keypoints saved in: {output_dir}")
pose.close()

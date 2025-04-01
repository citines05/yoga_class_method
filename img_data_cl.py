import os
import glob
import cv2
import mediapipe as mp

# Main image directory
image_base_dir = "dataset"  # replace with the actual dataset path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.75)

# Function to check if a person is present in the image
def is_person_in_image(image_path, confidence_threshold=0.3):
    """
    Checks if there is a person in the image based on keypoints detected by MediaPipe.
    :param image_path: Path to the image.
    :param confidence_threshold: Confidence threshold to consider valid keypoints.
    :return: False if no person is detected or if the image is corrupted.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"***Error loading image: {image_path}")
            return False

        # Convert to RGB (required for MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose
        results = pose.process(image_rgb)

        # Check if keypoints were detected
        if not results.pose_landmarks:
            return False

        # Check if the confidence threshold is met
        # confidences = [lm.visibility for lm in results.pose_landmarks.landmark]
        # if max(confidences) < confidence_threshold:
        #     return False

        return True

    except Exception as e:
        print(f"***Error processing image {image_path}: {e}")
        return False

# Collect image files from all subdirectories
image_paths = (glob.glob(image_base_dir + "/**/*.jpg", recursive=True) 
               + glob.glob(image_base_dir + "/**/*.png", recursive=True))

for img_path in image_paths:
    # Check if the image is valid and contains a person
    if not is_person_in_image(img_path):
        # Remove the invalid image
        print(f"---Removing invalid image: {img_path}")
        os.remove(img_path)

# Close MediaPipe
pose.close()
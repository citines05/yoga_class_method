import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Mapping of 82 classes to 20 superclasses
group_mapping = {
    # Standing - Straight
    "Eagle_Pose_or_Garudasana_": "Standing_Straight",
    "Tree_Pose_or_Vrksasana_": "Standing_Straight",
    "Chair_Pose_or_Utkatasana_": "Standing_Straight",

    # Standing - Forward bend
    "Standing_Forward_Bend_pose_or_Uttanasana_": "Standing_Forward_Bend",
    "Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_": "Standing_Forward_Bend",
    "Dolphin_Pose_or_Ardha_Pincha_Mayurasana_": "Standing_Forward_Bend",
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_": "Standing_Forward_Bend",
    "Intense_Side_Stretch_Pose_or_Parsvottanasana_": "Standing_Forward_Bend",

    # Standing - Side bend
    "Half_Moon_Pose_or_Ardha_Chandrasana_": "Standing_Side_Bend",
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_": "Standing_Side_Bend",
    "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_": "Standing_Side_Bend",
    "Gate_Pose_or_Parighasana_": "Standing_Side_Bend",
    "Warrior_I_Pose_or_Virabhadrasana_I_": "Standing_Side_Bend",
    "Viparita_virabhadrasana_or_reverse_warrior_pose": "Standing_Side_Bend",
    "Low_Lunge_pose_or_Anjaneyasana_": "Standing_Side_Bend",

    # Standing - Others
    "Warrior_II_Pose_or_Virabhadrasana_II_": "Standing_Others",
    "Warrior_III_Pose_or_Virabhadrasana_III_": "Standing_Others",
    "Lord_of_the_Dance_Pose_or_Natarajasana_": "Standing_Others",
    "Standing_big_toe_hold_pose_or_Utthita_Padangusthasana": "Standing_Others",
    "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_": "Standing_Others",

    # Sitting - Normal1 (legs in front)
    "Sitting pose 1 (normal)": "Sitting_Normal1",
    "Bound_Angle_Pose_or_Baddha_Konasana_": "Sitting_Normal1",
    "Garland_Pose_or_Malasana_": "Sitting_Normal1",
    "Staff_Pose_or_Dandasana_": "Sitting_Normal1",
    "Noose_Pose_or_Pasasana_": "Sitting_Normal1",

    # Sitting - Normal2 (legs behind)
    "Virasana_or_Vajrasana": "Sitting_Normal2",
    "Cow_Face_Pose_or_Gomukhasana_": "Sitting_Normal2",
    "Bharadvaja's_Twist_pose_or_Bharadvajasana_I_": "Sitting_Normal2",
    "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_": "Sitting_Normal2",

    # Sitting - Split
    "Split pose": "Sitting_Split",
    "Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_": "Sitting_Split",

    # Sitting - Forward bend
    "Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_": "Sitting_Forward_Bend",
    "Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_": "Sitting_Forward_Bend",
    "Seated_Forward_Bend_pose_or_Paschimottanasana_": "Sitting_Forward_Bend",
    "Tortoise_Pose": "Sitting_Forward_Bend",

    # Sitting - Twist
    "Akarna_Dhanurasana": "Sitting_Twist",
    "Heron_Pose_or_Krounchasana_": "Sitting_Twist",
    "Rajakapotasana": "Sitting_Twist",

    # Balancing - Front
    "Crane_(Crow)_Pose_or_Bakasana_": "Balancing_Front",
    "Shoulder-Pressing_Pose_or_Bhujapidasana_": "Balancing_Front",
    "Cockerel_Pose": "Balancing_Front",
    "Scale_Pose_or_Tolasana_": "Balancing_Front",
    "Firefly_Pose_or_Tittibhasana_": "Balancing_Front",

    # Balancing - Side
    "Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_": "Balancing_Side",
    "Eight-Angle_Pose_or_Astavakrasana_": "Balancing_Side",
    "Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II": "Balancing_Side",
}

# Input and output directories
input_dir = "kp_ds_aug_4" 
output_dir = "kp_ds_20_aug_4"

# Check if the output directory already exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Copy files while respecting the new structure
for split in ["train", "test", "valid"]:
    input_split_dir = os.path.join(input_dir, split)
    output_split_dir = os.path.join(output_dir, split)

    if not os.path.exists(output_split_dir):
        os.makedirs(output_split_dir)

    # List all original classes with tqdm
    for original_class in tqdm(os.listdir(input_split_dir), desc=f"Processing {split}"):
        original_class_dir = os.path.join(input_split_dir, original_class)

        if not os.path.isdir(original_class_dir):
            continue
        
        # Check if the class is mapped
        if original_class not in group_mapping:
            print(f"The class {original_class} is not mapped! Please check the dictionary.")
            continue
        
        # Identify the corresponding superclass
        super_class = group_mapping.get(original_class, "Others")
        super_class_dir = os.path.join(output_split_dir, super_class)

        if not os.path.exists(super_class_dir):
            os.makedirs(super_class_dir)

        # List all files in the original class with tqdm
        for file in tqdm(os.listdir(original_class_dir), desc=f"Copying {original_class}", leave=False):
            src_file = os.path.join(original_class_dir, file)
            
            # Ensure a unique filename
            unique_file_name = f"{original_class[:5]}_{file}"
            dst_file = os.path.join(super_class_dir, unique_file_name)
            
            shutil.copy(src_file, dst_file)

print("Reorganization complete. Data is stored in the directory:", output_dir)

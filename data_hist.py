import pathlib
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def count_files_in_all_splits(dataset_path):
    """
    Counts the number of files in each class for the train, test, and valid directories.

    :param dataset_path: Path to the dataset (main directory).
    :return: Dictionary with the class name as the key and the number of files per split as values.
    """
    splits = ['train', 'test', 'valid']  # Main subdirectories
    class_counts = defaultdict(lambda: {'train': 0, 'test': 0, 'valid': 0})

    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Subdirectory {split} not found in {dataset_path}")
            continue

        for class_name in os.listdir(split_path):
            class_path = split_path / class_name
            if class_path.is_dir():
                # Count the files in the class folder
                file_count = len([f for f in class_path.iterdir() if f.is_file()])
                class_counts[class_name][split] = file_count

    return class_counts

# Path to the dataset
dataset_path = pathlib.Path('dataset')  # Modify to the correct path

# Count files in each class
class_file_counts = count_files_in_all_splits(dataset_path)

# Map classes to numbered names
class_name_map = {cls: f"Class {i}" for i, cls in enumerate(class_file_counts)}

# Generate histograms for each split with classes sorted in descending order by file count
for split in ['train', 'test', 'valid']:
    # Sort classes alphabetically to get the mapping
    sorted_class_names = sorted(class_file_counts.keys())  # Alphabetical order
    class_name_map = {cls: f"Class {i}" for i, cls in enumerate(sorted_class_names)}  # Numeric mapping

    # Sort classes by file count in the current subset
    sorted_classes_by_count = sorted(class_file_counts.items(), key=lambda x: x[1][split], reverse=True)

    # Map labels to the format "Class X (real_name)"
    classes = [class_name_map[cls] for cls, _ in sorted_classes_by_count]
    file_counts = [counts[split] for _, counts in sorted_classes_by_count]

    # Generate the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(classes, file_counts, edgecolor='black', alpha=0.7)
    plt.xlabel("Classes")
    plt.ylabel(f"Number of Files ({split.capitalize()})")
    plt.title(f"File Distribution by Class (after preprocessing) - {split.capitalize()}")
    plt.xticks(rotation=90)  # Rotate class names for better readability
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

sorted_class_names = sorted(class_file_counts.keys())  # Sort real class names alphabetically
class_name_map = {cls: f"Class {i}" for i, cls in enumerate(sorted_class_names)}

# Display total counts with numbered and real names in alphabetical order
print("\nTotal count per class with numbered and real names (alphabetical order):")
for class_name in sorted_class_names:
    mapped_name = class_name_map[class_name]
    counts = class_file_counts[class_name]
    total = sum(counts.values())
    print(f"{mapped_name} ({class_name}): Train={counts['train']}, Test={counts['test']}, Valid={counts['valid']}, Total={total}")
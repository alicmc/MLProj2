import os
from collections import Counter

datasets = {
    "train": r"sawit/data/labels/VOC_format",
    "val": r"sawit/data/labels/VOC_format",
    "test": r"sawit/data/labels/VOC_format"
}

class_map = {
    "Big_mammal": 0,
    "Bird": 1,
    "Frog": 2,
    "Lizard": 3,
    "Scorpion": 4,
    "Small_mammal": 5,
    "Spider": 6
}

for split, path in datasets.items():
    class_counts = Counter()
    total_files = 0

    for file in os.listdir(path):
        if file.endswith(".txt"):
            total_files += 1
            file_path = os.path.join(path, file)
            with open(file_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_name = parts[0]              # e.g., "Big_mammal"
                    cls_idx = class_map[cls_name]    # convert to int
                    class_counts[cls_idx] += 1

    print(f"\n=== {split.upper()} SET ===")
    print(f"Total label files: {total_files}")
    print("Class distribution:")
    for cls_name, idx in class_map.items():
        print(f"  {cls_name} ({idx}): {class_counts[idx]} samples")

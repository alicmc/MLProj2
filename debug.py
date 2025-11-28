from dataset import DetectionAsClassificationDataset
import os

# Your class map
class_map = {
    'Big_mammal':0, 'Bird':1, 'Frog':2,
    'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
}

# Load dataset
ds = DetectionAsClassificationDataset(
    img_dir="sawit/data/images/train",
    label_dir=".\sawit\data\labels\VOC_format",
    transforms=None,
    class_map=class_map
)

print(f"Total samples: {len(ds)}")
print("---- Checking first 50 samples ----")

# We will access file paths by reconstructing from ds.files list
# Many PyTorch Datasets store paths in .files or similar attributes.
# If yours uses something else, we'll detect it.

# Try common attribute names:
possible_attrs = ["files", "images", "img_paths", "img_list", "image_files"]
attr = None
for a in possible_attrs:
    if hasattr(ds, a):
        attr = a
        break

if attr is None:
    print("\nERROR: Your dataset class does NOT expose image file paths.\n"
          "I need to see your dataset.py implementation.\n")
    exit()

files = getattr(ds, attr)

for i in range(100):
    img_path = files[i]
    base = os.path.basename(img_path)
    txt_name = os.path.splitext(base)[0] + ".txt"
    label_path = os.path.join(".\sawit\data\labels\VOC_format", txt_name)

    # Read raw label file text
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            raw = f.read().strip()
    else:
        raw = "<FILE MISSING>"

    img, label = ds[i]
    print(f"{i:02d} | label={label} | raw='{raw}' | label_file={txt_name}")

import os
from PIL import Image
from torch.utils.data import Dataset

class DetectionAsClassificationDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None, class_map=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.class_map = class_map

        # Collect image paths
        self.img_paths = []
        exts = ('.jpg', '.jpeg', '.png')

        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(exts):
                    self.img_paths.append(os.path.join(root, f))

        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def _parse_simple_label(self, label_path):
        """
        Example label:
        Big_mammal 1073 8 1626 1003
        """
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
            if len(line) == 0:
                return None

            parts = line.split()
            class_name = parts[0]

            if self.class_map and class_name in self.class_map:
                return self.class_map[class_name]
            else:
                return None  # unknown class
        except:
            return None

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Find matching label file
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")

        # Parse label
        label = self._parse_simple_label(label_path)

        if label is None:
            label = 0  # fallback, but should rarely happen

        if self.transforms:
            img = self.transforms(img)

        return img, label

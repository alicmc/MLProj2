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

    def _parse_label_with_box(self, label_path):
        """
        Format: class_name x_min y_min x_max y_max
        """
        try:
            with open(label_path, "r") as f:
                line = f.readline().strip()

            if not line:
                return None, None

            parts = line.split()
            class_name = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:])

            if self.class_map and class_name in self.class_map:
                cls = self.class_map[class_name]
            else:
                print(class_name)
                print(self.class_map)
                return None, None

            return cls, (x1, y1, x2, y2)

        except:
            return None, None


    # def __getitem__(self, idx):
    #     img_path = self.img_paths[idx]

    #     # Load image
    #     img = Image.open(img_path).convert("RGB")

    #     # Find matching label file
    #     base = os.path.splitext(os.path.basename(img_path))[0]
    #     label_path = os.path.join(self.label_dir, base + ".txt")

    #     # Parse label
    #     label = self._parse_simple_label(label_path)

    #     if label is None:s
    #         label = 0  # fallback, but should rarely happen

    #     if self.transforms:
    #         img = self.transforms(img)

    #     return img, label

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.normpath(os.path.join(self.label_dir, base + ".txt"))


        cls, box = self._parse_label_with_box(label_path)

        if cls is None or box is None:
            print(cls, box)
            raise ValueError(f"Invalid label file: {label_path}")

        x1, y1, x2, y2 = box

        # Crop to bounding box
        crop = img.crop((x1, y1, x2, y2))

        if self.transforms:
            crop = self.transforms(crop)

        return crop, cls

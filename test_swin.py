from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import timm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import constants
import os
from shutil import copy2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/test',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=test_transforms,
        class_map=constants.CLASS_MAP,
        return_path=True   # <-- MAKE SURE YOUR DATASET CAN RETURN PATHS
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_classes = len(constants.CLASS_MAP)
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
        global_pool='avg'
    )
    model.load_state_dict(torch.load("swin_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Testing on {len(test_dataset)} samples...")
    print(f"Using device: {device}\n")

    all_preds = []
    all_labels = []
    misclassified = []   # <-- store (path, true, pred)

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Track misclassifications
            for p, t, pr in zip(paths, labels.cpu().numpy(), preds.cpu().numpy()):
                if t != pr:
                    misclassified.append((p, t, pr))

    class_names = list(constants.CLASS_MAP.keys())

    print("="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # ==========================
    #  PER-CLASS ACCURACY
    # ==========================
    per_class_acc = []
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)

    for i, class_name in enumerate(class_names):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        per_class_acc.append(class_acc)
        print(f"{class_name:15s}: {class_correct:3d}/{class_total:3d} = {class_acc:.2f}%")

    # ==========================
    #  PLOT PER-CLASS ACCURACY
    # ==========================

    plt.figure(figsize=(12, 5))
    bars = plt.bar(class_names, per_class_acc)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45, ha='right')

    # Annotate bar values
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{acc:.1f}%", 
                 ha='center', va='bottom', fontsize=9)

    os.makedirs("charts", exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("charts", "per_class_accuracy.png"), dpi=300)
    plt.show()


    # ==========================
    #  SAVE MISCLASSIFIED IMAGES
    # ==========================

    print("\nSaving misclassification data...")

    df = pd.DataFrame(misclassified, columns=["path", "true_label", "predicted_label"])
    df["true_label_name"] = df["true_label"].apply(lambda x: class_names[x])
    df["predicted_label_name"] = df["predicted_label"].apply(lambda x: class_names[x])
    df.to_csv("misclassified.csv", index=False)

    print(f"Misclassified samples saved to misclassified.csv ({len(df)} samples)")

    # Optional: copy images into folder
    
    out_dir = "misclassified"
    os.makedirs(out_dir, exist_ok=True)
    for p, t, pr in misclassified:
        filename = os.path.basename(p)
        copy2(p, os.path.join(out_dir, f"true_{class_names[t]}__pred_{class_names[pr]}__{filename}"))
    

    print("\nDone!")


if __name__ == "__main__":
    main()

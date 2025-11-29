from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import timm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    class_map = {
        'Big_mammal':0, 'Bird':1, 'Frog':2,
        'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
    }

    test_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/test',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=test_transforms,
        class_map=class_map
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_classes = len(class_map)
    model = timm.create_model("swin_base_patch4_window7_224",
                            pretrained=False,
                            num_classes=num_classes,
                            global_pool='avg')
    model.load_state_dict(torch.load("best_swin_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Testing on {len(test_dataset)} samples...")
    print(f"Using device: {device}\n")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(all_labels, all_preds, target_names=list(class_map.keys())))
    
    # Confusion Matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Calculate and display per-class accuracy
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)
    class_names = list(class_map.keys())
    for i, class_name in enumerate(class_names):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f"{class_name:15s}: {class_correct:3d}/{class_total:3d} = {class_acc:.2f}%")
    
    # Overall accuracy
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    overall_acc = 100 * total_correct / total_samples
    print(f"\n{'Overall':15s}: {total_correct:3d}/{total_samples:3d} = {overall_acc:.2f}%")
    
    # Visualize Confusion Matrix
    print("\n" + "="*80)
    print("Generating confusion matrix visualization...")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    
    plt.title('Confusion Matrix - Swin Transformer', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the confusion matrix
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

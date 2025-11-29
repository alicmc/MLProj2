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
    
    # Separate lists excluding Frog class
    all_preds_no_frog = []
    all_labels_no_frog = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Store all predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store predictions excluding Frog samples
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if label != class_map['Frog']:  # Exclude Frog from ground truth
                    all_preds_no_frog.append(pred)
                    all_labels_no_frog.append(label)

    # Full results (with Frog)
    print("="*80)
    print("FULL CLASSIFICATION REPORT (ALL CLASSES)")
    print("="*80)
    print(classification_report(all_labels, all_preds, target_names=list(class_map.keys())))
    
    # Results without Frog class
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (EXCLUDING FROG CLASS)")
    print("="*80)
    
    # Create class names without Frog
    class_names_no_frog = [name for name in class_map.keys() if name != 'Frog']
    
    print(classification_report(all_labels_no_frog, all_preds_no_frog, 
                                target_names=class_names_no_frog,
                                labels=[class_map[name] for name in class_names_no_frog]))
    
    # Confusion Matrix without Frog
    print("\n" + "="*80)
    print("CONFUSION MATRIX (EXCLUDING FROG)")
    print("="*80)
    
    cm_no_frog = confusion_matrix(all_labels_no_frog, all_preds_no_frog,
                                   labels=[class_map[name] for name in class_names_no_frog])
    print(cm_no_frog)
    
    # Calculate and display per-class accuracy (excluding Frog)
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY (EXCLUDING FROG)")
    print("="*80)
    for i, class_name in enumerate(class_names_no_frog):
        class_total = cm_no_frog[i].sum()
        class_correct = cm_no_frog[i, i]
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        marker = "🎯" if class_name == 'Small_mammal' else "  "
        print(f"{marker} {class_name:15s}: {class_correct:3d}/{class_total:3d} = {class_acc:.2f}%")
    
    # Overall accuracy excluding Frog
    total_correct_no_frog = np.trace(cm_no_frog)
    total_samples_no_frog = cm_no_frog.sum()
    overall_acc_no_frog = 100 * total_correct_no_frog / total_samples_no_frog
    print(f"\n{'Overall (no Frog)':15s}: {total_correct_no_frog:3d}/{total_samples_no_frog:3d} = {overall_acc_no_frog:.2f}%")
    
    # Compare with full accuracy
    cm_full = confusion_matrix(all_labels, all_preds)
    total_correct_full = np.trace(cm_full)
    total_samples_full = cm_full.sum()
    overall_acc_full = 100 * total_correct_full / total_samples_full
    
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    print(f"With Frog:    {overall_acc_full:.2f}% ({total_samples_full} samples)")
    print(f"Without Frog: {overall_acc_no_frog:.2f}% ({total_samples_no_frog} samples)")
    diff = overall_acc_no_frog - overall_acc_full
    if diff > 0:
        print(f"📈 Accuracy IMPROVED by {diff:.2f}% without Frog class")
    else:
        print(f"📉 Accuracy DECREASED by {abs(diff):.2f}% without Frog class")
    
    # Visualize Confusion Matrix (without Frog)
    print("\n" + "="*80)
    print("Generating confusion matrix visualization (excluding Frog)...")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Full confusion matrix
    cm_full_display = confusion_matrix(all_labels, all_preds)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_full_display, 
                                    display_labels=list(class_map.keys()))
    disp1.plot(ax=ax1, cmap='Blues', values_format='d', colorbar=True)
    ax1.set_title('Full Confusion Matrix (All Classes)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_ylabel('True Label', fontsize=11)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Confusion matrix without Frog
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_no_frog, 
                                    display_labels=class_names_no_frog)
    disp2.plot(ax=ax2, cmap='Greens', values_format='d', colorbar=True)
    ax2.set_title('Confusion Matrix (Excluding Frog)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=11)
    ax2.set_ylabel('True Label', fontsize=11)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the confusion matrices
    plt.savefig('confusion_matrix_with_without_frog.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrices saved as 'confusion_matrix_with_without_frog.png'")
    
    # Show the plot
    plt.show()
    
    print("\n✓ Testing complete!")

if __name__ == "__main__":
    main()

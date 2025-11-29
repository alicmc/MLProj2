"""
Analyze what Frogs are being misclassified as in the test set
"""
from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import timm
from collections import Counter
import numpy as np

class_map = {
    'Big_mammal':0, 'Bird':1, 'Frog':2,
    'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
}
class_names = {v: k for k, v in class_map.items()}

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

print("="*80)
print("FROG MISCLASSIFICATION ANALYSIS")
print("="*80)

# Track Frog-specific errors
frog_id = class_map['Frog']
frog_predictions = []
frog_confidences = []
misclassified_as = Counter()

all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)
        
        # Store all predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())
        
        # Track Frog-specific errors
        for label, pred, conf in zip(labels, preds, confidences):
            if label.item() == frog_id:
                frog_predictions.append(pred.item())
                frog_confidences.append(conf.item())
                if pred.item() != frog_id:
                    misclassified_as[pred.item()] += 1

# Calculate Frog accuracy
frog_correct = sum(1 for p in frog_predictions if p == frog_id)
frog_total = len(frog_predictions)
frog_acc = 100 * frog_correct / frog_total if frog_total > 0 else 0

print(f"\nFROG TEST SET PERFORMANCE:")
print(f"  Total Frogs: {frog_total}")
print(f"  Correctly classified: {frog_correct}")
print(f"  Accuracy: {frog_acc:.2f}%")
print(f"  Average confidence: {np.mean(frog_confidences):.3f}")

print(f"\nFROGS MISCLASSIFIED AS:")
total_errors = frog_total - frog_correct
for class_id, count in misclassified_as.most_common():
    percentage = 100 * count / total_errors if total_errors > 0 else 0
    print(f"  {class_names[class_id]:15s}: {count:4d} ({percentage:5.1f}% of errors)")

# Check if other classes are confused AS Frogs
print(f"\nOTHER CLASSES MISCLASSIFIED AS FROG:")
confused_as_frog = Counter()
for true_label, pred in zip(all_labels, all_preds):
    if pred == frog_id and true_label != frog_id:
        confused_as_frog[true_label] += 1

for class_id, count in confused_as_frog.most_common():
    print(f"  {class_names[class_id]:15s}: {count:4d} samples")

# Calculate what percentage of "Frog" predictions are actually correct
frog_pred_count = sum(1 for p in all_preds if p == frog_id)
frog_precision = 100 * frog_correct / frog_pred_count if frog_pred_count > 0 else 0
print(f"\nFROG PRECISION: {frog_precision:.2f}%")
print(f"  (Out of {frog_pred_count} 'Frog' predictions, {frog_correct} were actually Frogs)")

# Confidence analysis
print(f"\nCONFIDENCE ANALYSIS:")
correct_frog_confidences = [conf for pred, conf in zip(frog_predictions, frog_confidences) if pred == frog_id]
wrong_frog_confidences = [conf for pred, conf in zip(frog_predictions, frog_confidences) if pred != frog_id]

if correct_frog_confidences:
    print(f"  Correct Frog predictions: avg confidence = {np.mean(correct_frog_confidences):.3f}")
if wrong_frog_confidences:
    print(f"  Wrong Frog predictions: avg confidence = {np.mean(wrong_frog_confidences):.3f}")
    print(f"  → Model is {'confident' if np.mean(wrong_frog_confidences) > 0.7 else 'uncertain'} when making mistakes")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

# Generate recommendations based on analysis
top_confusion = misclassified_as.most_common(1)[0] if misclassified_as else None
if top_confusion:
    confused_class = class_names[top_confusion[0]]
    print(f"\n1. Frogs are most confused with {confused_class}")
    print(f"   → Add more diverse Frog/...{confused_class} training examples")
    print(f"   → Increase data augmentation to better separate these classes")

if wrong_frog_confidences and np.mean(wrong_frog_confidences) > 0.7:
    print(f"\n2. Model is overconfident in wrong predictions")
    print(f"   → Increase label smoothing (currently 0.15, try 0.2)")
    print(f"   → Add more regularization (dropout, weight decay)")

if frog_precision < 80:
    print(f"\n3. Many non-Frogs are predicted as Frogs")
    print(f"   → Model has learned to over-predict Frog class")
    print(f"   → Consider reducing Frog class weight slightly")

print("\n")

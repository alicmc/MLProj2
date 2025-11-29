"""
Script to check class distribution in training and test datasets
"""
from dataset import DetectionAsClassificationDataset
from collections import Counter

class_map = {
    'Big_mammal':0, 'Bird':1, 'Frog':2,
    'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
}

# Check training set
train_dataset = DetectionAsClassificationDataset(
    img_dir='sawit/data/images/train',
    label_dir='./sawit/data/labels/VOC_format',
    transforms=None,
    class_map=class_map
)

# Check test set
test_dataset = DetectionAsClassificationDataset(
    img_dir='sawit/data/images/test',
    label_dir='./sawit/data/labels/VOC_format',
    transforms=None,
    class_map=class_map
)

# Count classes
print("="*60)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*60)

train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

train_counts = Counter(train_labels)
test_counts = Counter(test_labels)

# Reverse mapping
class_names = {v: k for k, v in class_map.items()}

print(f"\nTRAINING SET (Total: {len(train_dataset)} samples)")
print("-"*60)
for class_id in sorted(train_counts.keys()):
    count = train_counts[class_id]
    percentage = 100 * count / len(train_dataset)
    print(f"{class_names[class_id]:15s}: {count:5d} samples ({percentage:5.1f}%)")

print(f"\nTEST SET (Total: {len(test_dataset)} samples)")
print("-"*60)
for class_id in sorted(test_counts.keys()):
    count = test_counts[class_id]
    percentage = 100 * count / len(test_dataset)
    print(f"{class_names[class_id]:15s}: {count:5d} samples ({percentage:5.1f}%)")

print("\n" + "="*60)
print("IMBALANCE ANALYSIS")
print("="*60)

# Calculate imbalance ratio
max_train = max(train_counts.values())
min_train = min(train_counts.values())
print(f"\nTraining set imbalance ratio: {max_train/min_train:.2f}:1")

max_test = max(test_counts.values())
min_test = min(test_counts.values())
print(f"Test set imbalance ratio: {max_test/min_test:.2f}:1")

# Show which classes are underrepresented
print("\nUnderrepresented classes in TRAINING (< average):")
avg_train = sum(train_counts.values()) / len(train_counts)
for class_id, count in sorted(train_counts.items(), key=lambda x: x[1]):
    if count < avg_train:
        print(f"  {class_names[class_id]:15s}: {count:5d} (need {int(avg_train - count)} more)")

print("\n" + "="*60)

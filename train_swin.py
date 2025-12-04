from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to batch.
    Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    # -------------------------
    # HYPERPARAMETERS - Adjusted to reduce overfitting and prioritize classes
    # -------------------------
    EPOCHS = 25               # Reduced to prevent overfitting
    BATCH_SIZE = 16          # Reduce from 32 to 16 for better gradient updates
    LEARNING_RATE = 3e-5     # Even lower LR for stability
    WEIGHT_DECAY = 5e-4      # Increased regularization to reduce overfitting
    UNFREEZE_AFTER_EPOCH = 5 # Unfreeze earlier with fewer epochs
    MIXUP_ALPHA = 0.2        # Mixup augmentation for better generalization
    FROG_PRIORITY = 2.5      # Extra priority for Frog class (increased from 1.5)
    SMALL_MAMMAL_PRIORITY = 2.0  # Priority for Small_mammal
    
    # Class weighting - prioritize Frog (2) and Small_mammal (5)
    # Higher weights = model focuses more on these classes
    CLASS_WEIGHTS = {
        'Big_mammal': 1.0,
        'Bird': 1.0,
        'Frog': 2.0,          # 2x priority
        'Lizard': 1.0,
        'Scorpion': 1.0,
        'Small_mammal': 2.0,  # 2x priority
        'Spider': 1.0
    }
    
    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # -------------------------
    # 2. Enhanced Transforms with more augmentation
    # -------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),           # Resize larger
        transforms.RandomCrop((224, 224)),       # Then random crop
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(p=0.2),    # Vertical flip for animals
        transforms.RandomRotation(20),           # Increased rotation
        transforms.ColorJitter(                  # Stronger color augmentation
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        ),
        transforms.RandomAffine(                 # Affine transformations
            degrees=0,
            translate=(0.15, 0.15),              # More translation
            scale=(0.85, 1.15)                   # More scaling
        ),
        transforms.RandomGrayscale(p=0.1),       # Occasional grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))  # Random erasing for regularization
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    print("Transforms defined.")
    # -------------------------
    # 3. Dataset - Create separate train and validation datasets
    # -------------------------
    class_map = {
        'Big_mammal':0, 'Bird':1, 'Frog':2,
        'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
    }

    # Create full dataset without transforms first to get indices
    temp_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=None,
        class_map=class_map
    )
    
    # Split indices
    val_size = int(0.2 * len(temp_dataset))
    train_size = len(temp_dataset) - val_size
    train_indices, val_indices = random_split(
        range(len(temp_dataset)), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create separate datasets with appropriate transforms
    train_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=train_transforms,
        class_map=class_map
    )
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    
    val_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=val_transforms,  # No augmentation for validation
        class_map=class_map
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    # -------------------------
    # 3.5 Calculate class weights for balanced sampling (OPTIMIZED - FAST!)
    # -------------------------
    print("\nCalculating class distribution for balanced sampling...")
    
    # FASTEST: Extract labels directly from label files without loading images
    import os
    train_labels = []
    
    # Create temporary dataset just to get the image paths and parsing logic
    temp_full = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=None,
        class_map=class_map
    )
    
    # Only read label files (no image loading!)
    for idx in train_indices.indices:
        img_path = temp_full.img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(temp_full.label_dir, base + ".txt")
        label = temp_full._parse_simple_label(label_path)
        if label is not None:
            train_labels.append(label)
    
    print(f"Extracted {len(train_labels)} labels in seconds!")
    
    # Count samples per class
    class_counts = Counter(train_labels)
    print("\nTraining set class distribution:")
    class_names_reverse = {v: k for k, v in class_map.items()}
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = 100 * count / len(train_labels)
        print(f"  {class_names_reverse[class_id]:15s}: {count:5d} samples ({percentage:5.1f}%)")
    
    # Calculate sample weights (inverse frequency)
    # Classes with fewer samples get higher weight
    total_samples = len(train_labels)
    class_weights_for_sampling = {}
    for class_id, count in class_counts.items():
        class_weights_for_sampling[class_id] = total_samples / (len(class_counts) * count)
    
    # Create sample weights for each training sample
    sample_weights = [class_weights_for_sampling[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow oversampling of minority classes
    )
    
    print("\nApplying WeightedRandomSampler to balance classes during training")
    print("This will oversample Small_mammal and Spider, undersample Frog")
    
    # DataLoaders with balanced sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Dataset and DataLoader created. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    # -------------------------
    # 4. Model
    # -------------------------
    num_classes = len(class_map)
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
        global_pool='avg',
        drop_rate=0.4,        # Increased dropout to reduce overfitting
        drop_path_rate=0.3    # Increased stochastic depth
    )
    model = model.to(device)
    print("Model created with dropout=0.4 and drop_path=0.3 for stronger regularization.")
    # Freeze backbone initially (will unfreeze later)
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name

    # -------------------------
    # 5. Loss & optimizer with adaptive class weights
    # -------------------------
    # Calculate inverse frequency weights from actual distribution
    # This gives higher loss penalty for misclassifying rare classes
    max_count = max(class_counts.values())
    adaptive_weights = {}
    for class_id in range(num_classes):
        if class_id in class_counts:
            # Inverse frequency: rare classes get higher weight
            adaptive_weights[class_id] = max_count / class_counts[class_id]
        else:
            adaptive_weights[class_id] = 1.0
    
    # Apply extra multiplier for priority classes (Frog and Small_mammal)
    adaptive_weights[class_map['Frog']] *= FROG_PRIORITY
    adaptive_weights[class_map['Small_mammal']] *= SMALL_MAMMAL_PRIORITY
    
    # Convert to tensor
    weight_list = [adaptive_weights[i] for i in range(num_classes)]
    class_weights_tensor = torch.tensor(weight_list, dtype=torch.float32).to(device)
    
    print("\nAdaptive class weights (based on inverse frequency + priority):")
    for class_id in range(num_classes):
        weight_str = f"{weight_list[class_id]:.2f}x"
        if class_id == class_map['Frog']:
            weight_str += f" (PRIORITY: {FROG_PRIORITY}x boost for Frog)"
        elif class_id == class_map['Small_mammal']:
            weight_str += f" (PRIORITY: {SMALL_MAMMAL_PRIORITY}x boost)"
        print(f"  {class_names_reverse[class_id]:15s}: {weight_str}")
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,  # Adaptive weights for imbalanced classes
        label_smoothing=0.15          # Increased label smoothing to reduce overconfidence
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    print(f"Loss function and optimizer set. LR={LEARNING_RATE}")
    # -------------------------
    # 6. Training loop with early stopping - stops after 3 epochs of no improvement
    # -------------------------
    patience = 3              # Reduced patience to stop earlier
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(EPOCHS):
        # Unfreeze backbone layers after initial training
        if epoch == UNFREEZE_AFTER_EPOCH:
            print(f"\n>>> Unfreezing backbone layers at epoch {epoch+1}")
            for param in model.parameters():
                param.requires_grad = True
            # Recreate optimizer with all parameters and lower LR
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=LEARNING_RATE/10,  # Lower LR for fine-tuning
                weight_decay=WEIGHT_DECAY
            )
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Track per-class accuracy
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Apply mixup augmentation (helps with feature learning)
            if MIXUP_ALPHA > 0 and np.random.rand() > 0.5:  # Apply mixup 50% of the time
                images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                
                optimizer.zero_grad()
                preds = model(images)
                loss = mixup_criterion(criterion, preds, labels_a, labels_b, lam)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                
                # For mixup, we count accuracy based on primary label
                _, predicted = torch.max(preds, 1)
                total += labels_a.size(0)
                correct += (lam * (predicted == labels_a).float() + 
                           (1 - lam) * (predicted == labels_b).float()).sum().item()
            else:
                # Normal training without mixup
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy
                for label, pred in zip(labels, predicted):
                    class_total[label.item()] += 1
                    if label == pred:
                        class_correct[label.item()] += 1
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        
        # Calculate per-class training accuracy
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} Training Complete")
        print(f"Average Loss: {epoch_loss:.4f} | Overall Accuracy: {epoch_acc:.2f}%")
        print(f"\nPer-Class Training Accuracy:")
        for class_id in range(num_classes):
            if class_total[class_id] > 0:
                class_acc = 100 * class_correct[class_id] / class_total[class_id]
                class_name = class_names_reverse[class_id]
                marker = "🎯" if class_id in [class_map['Frog'], class_map['Small_mammal']] else "  "
                print(f"  {marker} {class_name:15s}: {class_acc:5.1f}% ({class_correct[class_id]}/{class_total[class_id]})")
        print(f"{'='*60}\n")
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Track per-class validation accuracy
        val_class_correct = {i: 0 for i in range(num_classes)}
        val_class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(preds, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Track per-class accuracy
                for label, pred in zip(labels, predicted):
                    val_class_total[label.item()] += 1
                    if label == pred:
                        val_class_correct[label.item()] += 1

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total  # Convert to percentage to match train_acc
        
        # Store metrics for plotting
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)

        print(f"\nValidation Results:")
        print(f"Overall - Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        print(f"\nPer-Class Validation Accuracy:")
        frog_acc = 0
        small_mammal_acc = 0
        for class_id in range(num_classes):
            if val_class_total[class_id] > 0:
                class_acc = 100 * val_class_correct[class_id] / val_class_total[class_id]
                class_name = class_names_reverse[class_id]
                marker = "🎯" if class_id in [class_map['Frog'], class_map['Small_mammal']] else "  "
                print(f"  {marker} {class_name:15s}: {class_acc:5.1f}% ({val_class_correct[class_id]}/{val_class_total[class_id]})")
                
                if class_id == class_map['Frog']:
                    frog_acc = class_acc
                elif class_id == class_map['Small_mammal']:
                    small_mammal_acc = class_acc
        
        # Check for overfitting
        acc_gap = epoch_acc - val_acc
        if acc_gap > 10:
            print(f"\n⚠️  Overfitting detected: Train acc is {acc_gap:.1f}% higher than Val acc")
        
        # Highlight priority class performance
        print(f"\n🎯 Priority Classes Performance:")
        print(f"   Frog: {frog_acc:.1f}% | Small_mammal: {small_mammal_acc:.1f}%")

        # ---- Early Stopping Check (based on validation accuracy) ----
        # Prioritize models that do well on both overall and Frog accuracy
        combined_score = val_acc + (frog_acc * 0.3)  # Bonus weight for Frog performance
        
        if combined_score > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_swin_model.pth")
            print(f"\n✓ Best model saved! (Val Acc: {val_acc:.2f}%, Frog Acc: {frog_acc:.1f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"\nNo improvement for {epochs_no_improve} epoch(s) (Best: {best_val_acc:.2f}%)")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break

    print("Training complete. Best model saved as best_swin_model.pth")
    
    # -------------------------
    # 8. Plot and Save Loss Curves
    # -------------------------
    print("\nGenerating loss curves...")
    
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs_range, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as 'training_curves.png'")
    plt.show()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"\nFinal Epoch Metrics:")
    print(f"  Training Loss: {train_losses[-1]:.4f} | Accuracy: {train_accs[-1]:.2f}%")
    print(f"  Validation Loss: {val_losses[-1]:.4f} | Accuracy: {val_accs[-1]:.2f}%")
    print(f"\nOverfitting Check:")
    final_gap = train_accs[-1] - val_accs[-1]
    if final_gap > 10:
        print(f"  ⚠️  Overfitting: {final_gap:.1f}% accuracy gap")
    else:
        print(f"  ✓ Good generalization: {final_gap:.1f}% accuracy gap")
    print(f"{'='*60}")
    print("\nNote: Model is optimized for Frog and Small_mammal classes (2x weight)")


if __name__ == "__main__":
    main()

from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import timm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    # -------------------------
    # HYPERPARAMETERS - Adjusted to reduce overfitting and prioritize classes
    # -------------------------
    EPOCHS = 12               # Reduced to prevent overfitting
    BATCH_SIZE = 16          # Reduce from 32 to 16 for better gradient updates
    LEARNING_RATE = 3e-5     # Even lower LR for stability
    WEIGHT_DECAY = 5e-4      # Increased regularization to reduce overfitting
    UNFREEZE_AFTER_EPOCH = 4 # Unfreeze earlier with fewer epochs
    
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
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
    # 5. Loss & optimizer with class weights
    # -------------------------
    # Convert class weights to tensor
    weight_list = [CLASS_WEIGHTS[cls] for cls in sorted(class_map.keys(), key=lambda x: class_map[x])]
    class_weights_tensor = torch.tensor(weight_list, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,  # Prioritize Frog and Small_mammal
        label_smoothing=0.15          # Increased label smoothing to reduce overconfidence
    )
    print(f"Class weights applied: {dict(zip(sorted(class_map.keys(), key=lambda x: class_map[x]), weight_list))}")
    
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
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
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
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} complete")
        print(f"Average Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(preds, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total  # Convert to percentage to match train_acc
        
        # Store metrics for plotting
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        print(f"Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Check for overfitting
        acc_gap = epoch_acc - val_acc
        if acc_gap > 10:
            print(f"⚠️  Overfitting detected: Train acc is {acc_gap:.1f}% higher than Val acc")
        
        # Warning for suspicious metrics
        if val_loss < 0.01:
            print("⚠️  WARNING: Validation loss is extremely low! This may indicate:")
            print("    - Data leakage between train/val sets")
            print("    - Very small validation set")
            print("    - Model memorization")

        # ---- Early Stopping Check (based on validation accuracy) ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_swin_model.pth")
            print(f"✓ Best model saved (Val Acc: {best_val_acc:.2f}%, Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
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

from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import timm
import torch
import torch.nn as nn
import constants
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

BATCH_SIZE = 16       
WEIGHT_DECAY = 1e-4      
UNFREEZE_AFTER_EPOCH = 5 

def train_model(validation=False, epochs=12, learning_rate=3e-4):          
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Defining transforms to augment test set
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),           # Resize larger
        transforms.RandomCrop((224, 224)),       # Then random crop
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(p=0.2),    # Vertical flip for animals
        transforms.RandomRotation(15),           # Random rotation
        transforms.ColorJitter(                  # Color augmentation
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(                 # Affine transformations
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    print("Transforms defined.")

    full_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=train_transforms,
        class_map=constants.CLASS_MAP
    )

    if validation:
        # Split dataset into train/val
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size]
        )

        # ---- Build sampler ONLY from training labels ----
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_count = np.bincount(train_labels)
        weights = 1.0 / class_count
        sample_weights = [weights[l] for l in train_labels]

        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # ---- Apply validation transforms safely ----
        val_dataset.dataset.transforms = val_transforms

        # ---- Dataloaders ----
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=train_sampler,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        print(f"Dataset and DataLoader created. Training samples: {len(train_dataset)}")

    else:
        # ---- No validation mode ----
        # Build all-samples sampler
        labels = [full_dataset[i][1] for i in range(len(full_dataset))]
        class_count = np.bincount(labels)
        weights = 1.0 / class_count
        sample_weights = [weights[l] for l in labels]

        full_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=full_sampler,
            num_workers=4
        )
        print(f"Dataset and DataLoader created. Training samples: {len(full_dataset)}")

    # 4. Model
    num_classes = len(constants.CLASS_MAP)
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
        global_pool='avg'
    )
    model = model.to(device)
    print("Model created.")
    # Freeze backbone initially (will unfreeze later)
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name

    # 5. Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    print(f"Loss function and optimizer set. LR={learning_rate}")

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Validation training loop
    for epoch in range(epochs):
        # Unfreeze backbone layers after initial training
        if epoch == UNFREEZE_AFTER_EPOCH:
            print(f"\n>>> Unfreezing backbone layers at epoch {epoch+1}")
            for param in model.parameters():
                param.requires_grad = True
            # Recreate optimizer with all parameters and lower LR
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate/10,  # Lower LR for fine-tuning
                weight_decay=WEIGHT_DECAY
            )
            # **Create a new scheduler linked to the new optimizer**
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
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
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} complete")
        print(f"Average Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")
        print(f"{'='*60}\n")
        
        model.eval()
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}")

        # Validation
        if(validation):
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
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc * 100)   # convert to percent if you want

            print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Update learning rate based on loss
        if validation:
            scheduler.step(val_loss)  # step on validation loss
        else:
            scheduler.step(epoch_loss)  # only if no val set exists


    torch.save(model.state_dict(), "swin_model.pth")
    print("Training complete. Model saved as swin_model.pth")

    # Plot and Save Loss Curves for Hyperparameter Tuning
    if validation:
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

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        plt.savefig(os.path.join("charts", f"training_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
        print("Training curves saved as 'training_curves.png'")
        plt.show()
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"  Training Loss: {train_losses[-1]:.4f} | Accuracy: {train_accs[-1]:.2f}%")
        print(f"  Validation Loss: {val_losses[-1]:.4f} | Accuracy: {val_accs[-1]:.2f}%")


if __name__ == "__main__":
    # This is what we used to train learning rate hyperparameter
    # lrs_to_test = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 3e-4]
    # for lr in lrs_to_test:
    #     print(f"Testing LR = {lr}")
    #     train_model(validation=True, epochs=3, learning_rate=lr)

    train_model()
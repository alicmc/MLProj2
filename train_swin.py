from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
import torch
import torch.nn as nn


def main():
    # -------------------------
    # HYPERPARAMETERS - Adjust these to improve accuracy
    # -------------------------
    EPOCHS = 20              # Increase from 5 to 20 for better training
    BATCH_SIZE = 16          # Reduce from 32 to 16 for better gradient updates
    LEARNING_RATE = 5e-5     # Lower learning rate for fine-tuning
    WEIGHT_DECAY = 1e-4      # Regularization strength
    UNFREEZE_AFTER_EPOCH = 5 # Unfreeze backbone layers after this epoch
    
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
    print("Transforms defined.")
    # -------------------------
    # 3. Dataset
    # -------------------------
    class_map = {
        'Big_mammal':0, 'Bird':1, 'Frog':2,
        'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
    }

    train_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',  # fixed path
        transforms=train_transforms,
        class_map=class_map
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dataset and DataLoader created. Training samples: {len(train_dataset)}")
    # -------------------------
    # 4. Model
    # -------------------------
    num_classes = len(class_map)
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

    # -------------------------
    # 5. Loss & optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    print(f"Loss function and optimizer set. LR={LEARNING_RATE}")
    # -------------------------
    # 6. Training loop
    # -------------------------
    best_loss = float('inf')
    
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
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "swin_model_best.pth")
            print(f"✓ Best model saved (loss: {best_loss:.4f})")

    # -------------------------
    # 7. Save final model
    # -------------------------
    torch.save(model.state_dict(), "swin_model_final.pth")
    print("\nFinal model saved as swin_model_final.pth")
    print(f"Best model saved as swin_model_best.pth (loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()

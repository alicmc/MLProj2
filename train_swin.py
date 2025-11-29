from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import timm
import torch
import torch.nn as nn


def main():
    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 2. Transforms
    # -------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # -------------------------
    # 3. Dataset
    # -------------------------
    class_map = {
        'Big_mammal':0, 'Bird':1, 'Frog':2,
        'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
    }

    full_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir='./sawit/data/labels/VOC_format',
        transforms=train_transforms,
        class_map=class_map
    )

    # Split dataset into train and validation
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transforms
    val_dataset.dataset.transforms = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

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

    # Freeze backbone
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name

    # -------------------------
    # 5. Loss & optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-4, weight_decay=1e-4)

    # -------------------------
    # 6. Training loop with early stopping
    # -------------------------
    num_epochs = 50
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

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
        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # ---- Early Stopping Check ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_swin_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete. Best model saved as best_swin_model.pth")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DetectionAsClassificationDataset  # your custom Dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 1. Transforms
    # -------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # -------------------------
    # 2. Dataset
    # -------------------------
    # Optional: map VOC class names to integers
    class_map = {'Big_mammal':0, 'Bird':1, 'Frog':2, 'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6}

    train_dataset = DetectionAsClassificationDataset(
        img_dir='sawit/data/images/train',
        label_dir=r'.\sawit\data\labels\VOC_format',
        #label_type='voc',  # or 'yolo' if using YOLO TXT labels
        transforms=train_transforms,
        class_map=class_map
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # -------------------------
    # 3. Model
    # -------------------------
    num_classes = len(class_map)
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
        global_pool='avg'
    )
    model = model.to(device)

    # Freeze backbone, train only head
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # -------------------------
    # 4. Loss and optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    # -------------------------
    # 5. Training loop
    # -------------------------
    for epoch in range(5):
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
        print(f"Epoch {epoch+1} complete, loss = {epoch_loss:.4f}")

    # -------------------------
    # 6. Save model
    # -------------------------
    torch.save(model.state_dict(), "swin_model.pth")
    print("Model saved as swin_model.pth")

if __name__ == "__main__":
    main()

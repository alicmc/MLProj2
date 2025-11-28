import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_data = datasets.ImageFolder("sawit/data/images/train", transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    num_classes = 7
    model = timm.create_model("swin_base_patch4_window7_224", 
                              pretrained=True,
                              num_classes=num_classes, 
                              global_pool='avg')
    #model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)

    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-4, weight_decay=1e-4)

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)
            # print(preds.shape)  # should be [batch_size, num_classes]
            # print(labels.shape) # should be [batch_size]
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} complete, loss = {epoch_loss:.4f}")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms

# Load a pretrained Swin Transformer
model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

# Check the architecture
#print(model)


train_data = datasets.ImageFolder(
    "sawit/data/images/train",
    transform=transforms.ToTensor()
)

num_classes = 10
model.head = nn.Linear(model.head.in_features, num_classes)

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)


for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} complete, loss={loss.item():.4f}")

from dataset import DetectionAsClassificationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import timm
from sklearn.metrics import classification_report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    class_map = {
        'Big_mammal':0, 'Bird':1, 'Frog':2,
        'Lizard':3, 'Scorpion':4, 'Small_mammal':5, 'Spider':6
    }

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

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=list(class_map.keys())))

if __name__ == "__main__":
    main()

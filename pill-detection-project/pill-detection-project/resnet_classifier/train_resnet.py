import sys
sys.path.append('/content/pill-detection-project')



from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
import torch
import os
from resnet_classifier.common import train_one_epoch, validate
from resnet_classifier.pill_crop_dataset import setup_dataset_with_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 73
epochs = 5
save_path = "resnet_classifier/weights/best_resnet.pth"

# 모델
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 데이터셋
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
class_names, train_loader, val_loader, class_weights = setup_dataset_with_weights(
    data_dir="/content/cropped_pills",
    transform=transform,
    val_ratio=0.2,
    batch_size=32,
    device=device
)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0.0
for epoch in range(1, epochs + 1):
    print(f"🔁 Epoch {epoch}/{epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"✅ Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"🔍 Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("📌 Best model saved.")

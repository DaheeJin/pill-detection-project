import sys
sys.path.append('/content/pill-detection-project')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from resnet_classifier.pill_crop_dataset import setup_bright_blur_dataloader


# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/content/cropped_pills"
resnet_path = "/content/pill-detection-project/resnet_classifier/weights/best_resnet.pth"# 기존 모델 경로
num_classes = 73
epochs = 5
save_path = "resnet_classifier/weights/best_bright_blur_finetuned.pth"

# 데이터 전처리
transform_bright_blur = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3),
        transforms.GaussianBlur(kernel_size=(3, 3))
    ], p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 데이터 로딩
class_names, train_loader, val_loader, class_weights = setup_bright_blur_dataloader(
    data_dir=data_dir,
    transform=transform_bright_blur,
    batch_size=32,
    val_ratio=0.2,
    device=device
)

# 모델 불러오기 및 세팅
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(resnet_path, map_location=device))
model.to(device)
model.train()

# fine-tune 설정: fc만 학습하려면 아래 주석 해제
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
best_val_acc = 0.0
for epoch in range(epochs):
    print(f"🔁 Epoch {epoch+1}/{epochs}")
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_acc = correct / total
    print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    val_acc = correct / total
    print(f"🔍 Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"📌 Best model saved to {save_path}")

print(f"🎯 Final Best Validation Accuracy: {best_val_acc:.4f}")

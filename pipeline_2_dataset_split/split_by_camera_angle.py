
import os
import shutil
from glob import glob
import yaml

# 경로 설정
img_dir = "/content/ai02_project/train_images" # train이미지가 있는 경로
label_dir = "/content/ai02_project/yolo_labels/train"
base_output = "/content/ai02_project/yolo_data"

# 출력 디렉토리 구성
for split in ["train", "val"]:
    os.makedirs(os.path.join(base_output, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_output, "labels", split), exist_ok=True)

# 이미지 파일 수집
img_paths = sorted(glob(os.path.join(img_dir, "*.png")))

# 분할 및 복사
for img_path in img_paths:
    img_name = os.path.basename(img_path)
    stem = os.path.splitext(img_name)[0]
    label_path = os.path.join(label_dir, stem + ".txt")

    # 파일명에 '_70_'이 포함되면 val로 분류
    split = "val" if "_70_" in img_name else "train"

    shutil.copy2(img_path, os.path.join(base_output, "images", split, img_name))
    if os.path.exists(label_path):
        shutil.copy2(label_path, os.path.join(base_output, "labels", split, stem + ".txt"))
    else:
        print(f"라벨 없음: {label_path}")

print("'_70_' 기준 이미지 및 라벨 분할 완료")

# data.yaml 생성
data_yaml = {
    'path': base_output,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 5,
    'names': ['size_1', 'size_2', 'size_3', 'size_4', 'size_5']
}

yaml_path = os.path.join(base_output, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

print(f"data.yaml 생성 완료: {yaml_path}")

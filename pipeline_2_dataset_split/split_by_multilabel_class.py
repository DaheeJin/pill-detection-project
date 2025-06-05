
import os
import shutil
from glob import glob
from collections import defaultdict, Counter
import random
import yaml

random.seed(42)

# 경로 설정
img_dir = "/content/ai02_project/train_images" # train이미지가 있는 경로
label_dir = "/content/ai02_project/yolo_labels/train"
base_output = "/content/ai02_project/yolo_data_multilabel"

# 출력 폴더 생성
for split in ["train", "val"]:
    os.makedirs(os.path.join(base_output, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_output, "labels", split), exist_ok=True)

# Step 1: 이미지별 포함 class 수집
image_classes_map = {}  # {stem: [class_id1, class_id2, ...]}
class_to_images = defaultdict(set)  # {class_id: set(stem)}

label_paths = sorted(glob(os.path.join(label_dir, "*.txt")))
for label_path in label_paths:
    stem = os.path.splitext(os.path.basename(label_path))[0]
    with open(label_path, "r") as f:
        lines = f.readlines()
    class_ids = [line.split()[0] for line in lines]
    image_classes_map[stem] = class_ids
    for cid in class_ids:
        class_to_images[cid].add(stem)

# Step 2: 클래스별 후보 뽑되, 전체 val 이미지 수는 전체의 20%로 제한
val_candidate_counter = Counter()
for cid, stems in class_to_images.items():
    n_candidate = max(1, int(0.4 * len(stems)))  # 넉넉하게 뽑아놓기
    sampled = random.sample(list(stems), n_candidate)
    val_candidate_counter.update(sampled)

# 중복을 고려해 val_set 선정 (val 비율 = 20%)
val_target_size = int(0.2 * len(image_classes_map))
sorted_candidates = sorted(val_candidate_counter.items(), key=lambda x: (-x[1], x[0]))  # 등장 많은 순
val_set = set([stem for stem, _ in sorted_candidates[:val_target_size]])


# Step 3: 파일 복사
all_stems = set(image_classes_map.keys())
train_set = all_stems - val_set

for split_set, split_name in [(train_set, "train"), (val_set, "val")]:
    for stem in split_set:
        img_path = os.path.join(img_dir, f"{stem}.png")
        label_path = os.path.join(label_dir, f"{stem}.txt")

        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(base_output, "images", split_name, f"{stem}.png"))
        else:
            print(f"이미지 없음: {img_path}")
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(base_output, "labels", split_name, f"{stem}.txt"))

print("클래스 기반 multilabel-aware soft stratified split 완료")

# Step 4: data.yaml 저장
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



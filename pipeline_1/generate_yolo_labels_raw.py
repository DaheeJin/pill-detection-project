
import os
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict

# 설정
json_root = "/content/train_annotations_2/annotations_add_넥시움정" # JSON 파일이 있는 폴더 경로
output_dir = "/content/ai02_project/yolo_labels_raw/train"
os.makedirs(output_dir, exist_ok=True)

# Step 1: 이미지 기준으로 JSON 그룹핑
json_paths = glob(os.path.join(json_root, "**", "*.json"), recursive=True)
image_to_jsons = defaultdict(list)

for json_path in json_paths:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        image_filename = data.get("images", [])[0]["file_name"]
        image_to_jsons[image_filename].append(json_path)
    except Exception as e:
        print(f"JSON 파싱 오류: {json_path} → {e}")
        continue

# Step 2: YOLO TXT 생성
for image_filename, json_list in tqdm(image_to_jsons.items(), desc="YOLO TXT 생성"):
    lines = []
    stem = os.path.splitext(image_filename)[0]

    for json_path in json_list:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            image_info = data.get("images", [])[0]
            img_w, img_h = image_info["width"], image_info["height"]

            for ann in data.get("annotations", []):
                class_id = str(ann["category_id"])  # annotation별로 category_id 사용
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w /= img_w
                h /= img_h
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        except Exception as e:
            print(f"어노테이션 오류: {json_path} → {e}")
            continue

    if lines:
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        with open(txt_path, "w") as f:
            f.write("
".join(lines))

print("Pipeline 1 완료: YOLO TXT 생성 (원본 class ID 사용)")


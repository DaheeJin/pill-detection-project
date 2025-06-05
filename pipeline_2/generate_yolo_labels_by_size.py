
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# 설정
json_root = "/content/train_annotations_2/annotations_add_넥시움정" # JSON 파일이 있는 폴더 경로
output_dir = "/content/yolo_labels/train"
os.makedirs(output_dir, exist_ok=True)

# Step 1: class_id → 면적 계산
class_area_map = {}
json_paths = glob(os.path.join(json_root, "**", "*.json"), recursive=True)

for json_path in tqdm(json_paths, desc="면적 수집"):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        image_info = data.get("images", [])[0]
        category_info = data.get("categories", [])[0]
        class_id = category_info["id"]

        if class_id not in class_area_map:
            long_len = image_info.get("leng_long", 0)
            short_len = image_info.get("leng_short", 0)
            area = long_len * short_len
            class_area_map[class_id] = area
    except:
        continue

# Step 2: 분위수 기반 size class ID 생성
areas = list(class_area_map.values())
bins = np.quantile(areas, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
bins = np.unique(bins)

class_size_map = {}
for cid, area in class_area_map.items():
    for i in range(len(bins) - 1):
        if bins[i] <= area <= bins[i + 1]:
            class_size_map[cid] = i
            break

# Step 3: 이미지 기준 중복 JSON 그룹핑
image_to_jsons = defaultdict(list)
for json_path in json_paths:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        image_filename = data.get("images", [])[0]["file_name"]
        image_to_jsons[image_filename].append(json_path)
    except:
        continue

# Step 4: YOLO TXT 생성
for image_filename, json_list in tqdm(image_to_jsons.items(), desc="YOLO TXT 생성"):
    lines = []
    stem = os.path.splitext(image_filename)[0]

    for json_path in json_list:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            image_info = data.get("images", [])[0]
            img_w, img_h = image_info["width"], image_info["height"]
            category_info = data.get("categories", [])[0]
            original_class_id = category_info["id"]

            if original_class_id not in class_size_map:
                continue
            size_class_id = class_size_map[original_class_id]

            for ann in data.get("annotations", []):
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w /= img_w
                h /= img_h
                lines.append(f"{size_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        except Exception as e:
            print(f"오류 in {json_path}: {e}")
            continue

    if lines:
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

print("Pipeline 2 완료: YOLO TXT 생성 완료 (클래스 면적 기반 재분배)")

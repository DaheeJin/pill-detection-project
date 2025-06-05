import os
import cv2
import json
import argparse
import torch
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from ultralytics import YOLO
from torchvision import models

def build_drug_map(json_root):
    drug_map = {}
    json_paths = glob(os.path.join(json_root, "**", "*.json"), recursive=True)

    for json_path in json_paths:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            images = data.get("images", [])
            categories = data.get("categories", [])
            if images and categories:
                drug_N = images[0].get("drug_N")
                category_id = categories[0].get("id")
                category_name = categories[0].get("name")

                if drug_N and category_id:
                    drug_map[drug_N] = (category_id, category_name)
        except Exception as e:
            print(f"⚠️ 오류 발생: {json_path} → {e}")
    return drug_map

def export_detection_classification_csv(image_dir, yolo_model, classifier_model, transform, device, class_names, drug_map, save_csv_path="results.csv"):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    results = []
    ann_id = 1

    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        try:
            image_id = int(base_name)
        except ValueError:
            print(f"❌ 숫자 파일명 아님: {filename}")
            continue

        orig_image = cv2.imread(img_path)
        yolo_result = yolo_model.predict(
            source=img_path,
            conf=0.3,
            imgsz=512,
            iou=0.5,
            agnostic_nms=True,
            save=False,
            verbose=False
        )[0]  # ← 단일 이미지 결과 추출

        bboxes = yolo_result[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            crop = orig_image[y1:y2, x1:x2]

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            input_tensor = transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = classifier_model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            pred_class = class_names[pred_idx.item()]
            category_id = drug_map.get(pred_class, [None])[0]

            results.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": int(x1),
                "bbox_y": int(y1),
                "bbox_w": int(w),
                "bbox_h": int(h),
                "score": round(conf.item(), 4)
            })
            ann_id += 1

    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f"✅ CSV 저장 완료: {save_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--yolo_weights", required=True)
    parser.add_argument("--resnet_weights", required=True)
    parser.add_argument("--drug_map_json_root", required=True)
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 모델 불러오기
    yolo_model = YOLO(args.yolo_weights)

    classifier_model = models.resnet18(weights=None)
    classifier_model.fc = torch.nn.Linear(classifier_model.fc.in_features, 73)
    classifier_model.load_state_dict(torch.load(args.resnet_weights, map_location=device))
    classifier_model.to(device)
    classifier_model.eval()

    # 클래스 이름 목록
    class_names = sorted(os.listdir("/content/cropped_pills"))


    # Transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # drug_map
    drug_map = build_drug_map(args.drug_map_json_root)

    # 실행
    export_detection_classification_csv(
        image_dir=args.image_dir,
        yolo_model=yolo_model,
        classifier_model=classifier_model,
        transform=transform,
        device=device,
        class_names=class_names,
        drug_map=drug_map,
        save_csv_path=args.output_csv
    )

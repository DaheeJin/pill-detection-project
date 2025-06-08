import os
import cv2
import argparse
import torch
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from ultralytics import YOLO
from torchvision import models

def export_detection_classification_csv(image_dir, yolo_model, classifier_model, transform, device, save_csv_path="results.csv"):
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
        if orig_image is None:
            print(f"❌ 이미지 로딩 실패: {img_path}")
            continue

        yolo_result = yolo_model.predict(
            source=img_path,
            conf=0.3,
            imgsz=512,
            iou=0.5,
            agnostic_nms=True,
            save=False,
            verbose=False
        )[0]

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

            # class_id (0~72) → category_id (고유값)
            category_id = class_to_category[class_id]


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
    parser.add_argument("--image_dir", required=True, help="YOLO 추론용 이미지 경로")
    parser.add_argument("--yolo_weights", required=True, help="YOLO 모델 가중치 경로")
    parser.add_argument("--resnet_weights", required=True, help="ResNet 분류 모델 가중치 경로")
    parser.add_argument("--output_csv", default="results.csv", help="결과 저장 CSV 경로")
    parser.add_argument("--device", default="cuda", help="cuda 또는 cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 모델 로드
    yolo_model = YOLO(args.yolo_weights)

    classifier_model = models.resnet18(weights=None)
    classifier_model.fc = torch.nn.Linear(classifier_model.fc.in_features, 73)
    classifier_model.load_state_dict(torch.load(args.resnet_weights, map_location=device))
    classifier_model.to(device)
    classifier_model.eval()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 실행
    export_detection_classification_csv(
        image_dir=args.image_dir,
        yolo_model=yolo_model,
        classifier_model=classifier_model,
        transform=transform,
        device=device,
        save_csv_path=args.output_csv
    )

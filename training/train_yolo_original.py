
# [A] 약한 증강 포함 학습 (흐리멍텅한 이미지 일반화 성능 확보 목적)


from ultralytics import YOLO

# 모델 초기화
model = YOLO("yolov8m.pt")

model.train(
    data="/content/ai02_project/yolo_data/data.yaml",
    epochs=20,
    imgsz=512,
    batch=16,
    name="v8m_original",

    # 증강 설정
    degrees=10,
    translate=0.1,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.1
)

print("[A] 약한 증강 데이터 학습 완료")

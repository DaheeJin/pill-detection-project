# Pill Detection & Classification Project

본 프로젝트는 다양한 알약이 포함된 이미지에서 YOLO 기반 탐지 모델과 ResNet 기반 분류 모델을 활용하여 **알약의 위치와 종류를 예측**하는 딥러닝 파이프라인입니다.

---

## 📌 프로젝트 개요

- **목표**: 이미지 내 알약을 탐지하고, 해당 알약의 클래스를 정확히 분류
- **모델 구조**:  
  - **YOLOv8**: 알약 객체 탐지 (bounding box 예측)  
  - **ResNet18**: 감지된 알약 영역을 crop하여 세부 분류
- **데이터 형식**: COCO JSON 기반 어노테이션, 이미지별 다수 알약 포함

---

## 🧪 실험 내용 요약

### 1️⃣ YOLO 탐지 모델
- 사용 모델: `YOLOv8m.pt` fine-tuning
- 학습 데이터: 클래스 불균형 보완 및 어노테이션 수작업 정제
- 결과: 알약 위치에 대한 bounding box 예측 수행

### 2️⃣ ResNet 분류 모델
- 베이스: `torchvision.models.resnet18(pretrained=True)`
- 입력 전처리: `224x224` resizing + Normalize
- 학습 전략:
  - 기본 학습: `/content/cropped_pills`로부터 클래스별 crop 이미지로 학습
  - 추가 학습: 밝기 및 blur 증강 데이터로 fine-tuning
- 체크포인트:
  - `best_resnet.pth`: 기본 학습 최적 모델
  - `best_bright_blur_finetuned.pth`: 증강 데이터 기반 fine-tuned 모델

### 3️⃣ Inference 파이프라인
- `YOLO` 탐지를 통해 알약 영역 crop
- `ResNet` 모델로 각 crop된 알약 분류
- `drug_map` (JSON 기반)으로 클래스 이름과 ID 매핑
- 최종 결과를 `submission.csv`로 저장

---

## 📁 폴더 구조

```

pill-detection-project/
├── resnet\_classifier/
│   ├── common.py                  # 학습 루프 함수
│   ├── pill\_crop\_dataset.py      # 데이터셋 로딩 및 class weight 설정
│   ├── train\_resnet.py           # ResNet 기본 학습
│   ├── finetune\_resnet.py        # 증강 이미지 기반 fine-tuning
│   ├── inference\_resnet.py       # YOLO + ResNet 파이프라인 추론
│   └── weights/
│       ├── best\_resnet.pth
│       └── best\_bright\_blur\_finetuned.pth
└── ...

````
## 실행 코드

!python /content/pill-detection-project/resnet_classifier/inference_resnet.py \
  --image_dir /content/ai02_project/test_images \  #개인 테스트 이미지 경로로 변경하기
  --yolo_weights "/content/runs/detect/v8m_class5_70val_r5t0.05v0.1_(1)/weights/best.pt" \  #개인 YOLOv8m 웨이트 저장 경로로 변경하기
  --resnet_weights /content/pill-detection-project/resnet_classifier/weights/best_resnet.pth \
  --drug_map_json_root /content/ai02_project/train_annotations_2 \  #개인 트레인 어노테이션 경로로 변경하기
  --output_csv /content/test_results.csv

---

## 🔧 실행 예시

```bash
# ResNet 모델 학습
python resnet_classifier/train_resnet.py

# Fine-tuning (밝기/블러 증강 데이터)
python resnet_classifier/finetune_resnet.py

# 최종 추론 및 CSV 저장
python resnet_classifier/inference_resnet.py     --yolo_weights <YOLO 모델 경로>     --image_dir /content/ai02_project/test_images     --output_csv test_predictions.csv
````

---

## 🧩 참고 사항

* YOLO 모델은 다른 팀원의 `.pt` 파일도 활용 가능
* 분류 클래스 수: 73개 (`ImageFolder` 디렉토리 기준)
* 어노테이션 JSON은 `drug_N` 기준으로 `category_id`, `dl_name`을 매핑함

---

## 🙋‍♀️ 작성자

* 다희 | KDT AI 엔지니어링 과정 수강

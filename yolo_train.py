from ultralytics import YOLO

# 1. 모델 초기화 (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l 등 선택 가능)
model = YOLO("yolov8n.pt")  # Pre-trained 모델 로드 (YOLOv8n)

# 2. 데이터셋 및 학습 설정
data_yaml = "data.yaml"  # 데이터셋 구성 파일
epochs = 50              # 학습 반복 횟수
batch_size = 16          # 배치 크기

# 3. 모델 학습
model.train(
    data=data_yaml,       # 데이터셋 경로
    epochs=epochs,        # 에포크 수
    batch=batch_size,     # 배치 크기
    imgsz=640,            # 이미지 크기
    workers=4,            # 워커 수
    save_period=5,        # 모델 저장 주기 (Epoch마다 저장)
    project="runs/train", # 결과 저장 경로
    name="yolov8_custom"  # 프로젝트 이름
)

# 4. 학습 결과 확인
results = model.val()  # 검증 결과 확인
print("Validation Results:", results)

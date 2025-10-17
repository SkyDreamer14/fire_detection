# Firedect - YOLOv8 Hand Detection (Webcam)

간단한 실시간 손 감지 데모입니다. 웹캠 영상을 받아 YOLOv8 모델로 손을 감지하고, 박스와 확률을 화면에 표시합니다.

## 설치 (Windows PowerShell)

```powershell
cd C:\Users\KWCH\code\firedect
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r Firedect\requirements.txt
```

## 실행

웹캠 인덱스가 0일 때:

```powershell
python Firedect\hand_cam.py --source 0
```

- 'q' 키로 종료합니다.
- 기본적으로 핸드 전용 가중치를 Hugging Face에서 자동으로 다운로드를 시도합니다. 실패 시 일반 COCO 모델(`yolov8n.pt`)로 동작하며, 이 경우 손이 잘 감지되지 않을 수 있습니다.

직접 가중치(.pt)를 지정하려면:

```powershell
python Firedect\hand_cam.py --weights path\to\hand_model.pt
```

추가 옵션:

- `--imgsz 640` 추론 입력 크기
- `--conf 0.25` confidence threshold
- `--device cpu` 또는 CUDA GPU가 있다면 `--device 0`
- `--pref n|s|m|l` 자동 다운로드 시 선호 모델 크기

## (선택) 직접 파인튜닝

손 데이터셋이 있다면 YOLOv8으로 간단히 학습할 수 있습니다. 데이터는 YOLO 포맷을 가정합니다.

예시 `data_hand.yaml`:

```yaml
path: ./dataset
train: images/train
val: images/val
nc: 1
names: [hand]
```

학습 예시:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data_hand.yaml', epochs=50, imgsz=640)
# 완료 후 runs/train/exp*/weights/best.pt 사용
```

학습이 끝나면:

```powershell
python Firedect\hand_cam.py --weights runs\train\exp\weights\best.pt
```

## 트러블슈팅

- 웹캠이 열리지 않음: `--source 1` 등 다른 인덱스를 시도하세요. 다른 앱이 카메라를 점유하지 않았는지 확인하세요.
- 창이 뜨지만 손이 검출되지 않음: 핸드 전용 가중치를 명시적으로 제공하세요(`--weights`).
- CUDA 사용: 적합한 PyTorch CUDA 휠이 필요합니다. `--device 0`로 지정하고, GPU가속이 안 된다면 CPU로 자동 폴백합니다.

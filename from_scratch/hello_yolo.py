"""
1. 라이브러리 임포트
2. 
"""

##### 1. 라이브러리 임포트 #####
import os
import sys
from ultralytics import YOLO
##################


image_path = "C:/Users/KWCH/code/firedect/Firedect/from_scratch/sunglass.jpg"

def main():
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다. 이름: {image_path}")
        sys.exit(1)
        return
    
    print("YOLO 모델 로드 중..")
    model = YOLO("yolov8n.pt")
    print("YOLO 모델 로드 완료")
    
    print("추론 중..")
    results = model.predict(source=image_path, verbose=False)
    print("추론 완료")
    
    result = results[0]
    
    print(f"{image_path}에 대한 객체 탐지 결과 {len(result.boxes)}개의 객체가 탐지 되었습니다.")
    
    if len(result.boxes) > 0:
        print(result.boxes)
        
        print(result.boxes.xyxy)
        
        print(result.boxes.conf)
        
        print(result.boxes.cls)
    
    print("객체 탐지 결과 출력 완료")
    print(result.names)

if __name__ == "__main__":
    main()
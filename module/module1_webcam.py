"""
0. 라이브러리 설치 (pip install opencv-python) # OpenCV 라이브러리 설치
1. 라이브러리 임포트
2. 웹캠 연결
3. 무한 루프 내에 웹캠 구현
3-1. 웹캠 읽기/읽은 여부 반환
3-2. 웹캠 프레임 표시
4. 종료 구현
4-1. 키 입력 대기
4-2. 키 입력 처리
5. 웹캠 연결 해제
"""

import cv2 
import sys

def webcam_connect():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠 연결 실패")
        sys.exit(1)
    
    window_name = "Real-time CCTV"
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("카메라 연결이 끊어졌습니다.")
                break
            
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료")
                break
    finally:
        print("종료")
        cap.release()
        print("웹캠 연결 해제")
        cv2.destoryAllWindows()
    
    return 0




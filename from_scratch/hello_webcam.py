"""
0. 라이브러리 설치 (pip install opencv-python) # OpenCV 라이브러리 설치
1. 라이브러리 임포트
2. 웹캠 연결
3. 무한 루프 내에 웹캠 구현
3-1. 웹캠 읽기/읽은 여부 반환
3-2. 웹캠 프레임 표시
4. 종료 구현
4-1. 키 입력 대기 및 처리   
5. 웹캠 연결 해제
"""

import cv2
import sys

def main():
    ###### 2. 웹캠 연결 ######
    cap = cv2.VideoCapture(0)
    # cv2.imread('image.png') 이렇게 하면 image.png 이미지를 읽어옴
    # cv2.imshow('image', image) 이렇게 하면 image 이미지를 표시함
    # cv2.waitKey(0) 이렇게 하면 키 입력을 기다림
    # cv2.destroyAllWindows() 이렇게 하면 모든 창을 닫음
    # cv2.release() 이렇게 하면 웹캠 연결을 해제함
    # cv2.imwrite('image.png', image) 이렇게 하면 image.png 이미지를 저장함
    # cv2.imencode('.png', image)[1].tofile(f) 이렇게 하면 image.png 이미지를 저장함
    # cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) 이렇게 하면 image.png 이미지를 읽어옴
    
    if not cap.isOpened():
        print("웹캠 연결 실패")
        sys.exit(1) # 파이썬 프로그램을 즉시 종료하고, 운영체제에 종료 코드 1을 반환함 
        
    window_name = "Hello Webcam"
    #########################
    
    ####### 3. 무한 루프 내에 웹캠 구현 #######
    try:
        while True:
            ####### 3-1. 웹캠 읽기/읽은 여부 반환 #######
            ret, frame = cap.read()
            # ret: frame 읽기 성공 여부, frame: 읽어온 영상 프레임
            if not ret: # 카메라 연결이 끊겼을 경우 종료
                print("카메라 연결이 끊겼습니다.")
                break
            #########################################################
            
            ####### 3-2. 웹캠 프레임 표시 #######   
            cv2.imshow(window_name, frame) # window_name 창에 frame 영상 표시
            #########################################################
            
            ####### 4-1. 키 입력 대기 및 처리 #######
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료")
                break
            #########################################################
    finally:
        ####### 5. 웹캠 연결 해제 #######
        print("종료")
        cap.release()
        print("웹캠 연결 해제")
        cv2.destroyAllWindows()
        
    
if __name__ == "__main__":
    main()

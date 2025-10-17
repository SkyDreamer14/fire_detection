from ultralytics import YOLO
import os
import requests
import cv2

def download_image(url, save_path):
    """
    주어진 URL에서 이미지를 다운로드하여 지정된 경로에 저장합니다.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 실패: {e}")
        return False

def main():
    """
    인터넷의 공개 데이터셋을 사용하여 사전 학습된 YOLOv8n 모델을 파인튜닝하고,
    학습된 모델을 사용하여 테스트 이미지에 대한 추론을 실행합니다.
    """
    # --- 사용자 설정 ---
    # 1) 학습에 사용할 공개 데이터셋 (COCO128) YAML URL
    #    로컬에 파일이 없어도 Ultralytics가 URL에서 자동으로 다운로드하여 사용합니다.
    dataset_yaml_url = (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/"
        "ultralytics/cfg/datasets/coco128.yaml"
    )
    
    # 2. 학습을 얼마나 진행할지 Epoch 수를 설정합니다. (실습용으로 낮게 설정)
    epochs = 25
    
    # 3. 학습 결과를 저장할 프로젝트 이름과 실행 이름을 지정합니다.
    project_name = "chess_detection"
    run_name = "finetune_online_data"

    # 4. 테스트에 사용할 이미지 URL
    test_image_url = "https://images.chesscomfiles.com/uploads/v1/images_files/1/000/276/808/cHJb2o3i.jpeg"
    test_image_filename = "chess_test_image.jpg"
    # --- --- --- --- ---

    # 1. 사전 학습된(pre-trained) YOLOv8n 모델을 로드합니다.
    print("사전 학습된 YOLOv8n 모델을 로드합니다.")
    model = YOLO('yolov8n.pt')

    # 2. 모델 파인튜닝을 시작합니다.
    # `data` 인자에 공개 데이터셋 YAML URL을 전달합니다.
    print("COCO128 공개 데이터셋으로 모델 파인튜닝을 시작합니다...")
    # 처음 실행 시 데이터셋/라벨을 다운로드하므로 시간이 걸릴 수 있습니다.
    results = model.train(
        data=dataset_yaml_url,
        epochs=epochs,
        imgsz=640,
        project=project_name,
        name=run_name,
        exist_ok=True,
        device=0
    )
    print("모델 학습이 완료되었습니다.")
    print(f"학습된 모델과 결과는 '{project_name}/{run_name}' 폴더에 저장되었습니다.")
    print("-" * 30)

    # 3. 학습된 모델로 성능을 확인합니다.
    best_model_path = os.path.join(project_name, run_name, 'weights/best.pt')
    
    if not os.path.exists(best_model_path):
        print(f"[오류] 학습된 모델 파일 '{best_model_path}'를 찾을 수 없습니다.")
        return
        
    # 4. 테스트를 위해 인터넷에서 이미지를 다운로드합니다.
    print(f"테스트 이미지 다운로드 중... URL: {test_image_url}")
    if not download_image(test_image_url, test_image_filename):
        return

    print("가장 성능이 좋았던 모델로 테스트 추론을 실행합니다...")
    # 파인튜닝된 모델을 로드합니다.
    trained_model = YOLO(best_model_path)

    # 다운로드한 테스트 이미지로 추론을 실행합니다.
    predict_results = trained_model.predict(source=test_image_filename)
    
    # 첫 번째 이미지의 추론 결과를 가져옵니다.
    result = predict_results[0]
    
    # 추론 결과를 시각화하여 파일로 저장합니다.
    output_filename = "fine_tuned_chess_result.jpg"
    result.save(filename=output_filename)
    
    print(f"추론 결과 이미지가 '{output_filename}'으로 저장되었습니다.")
    
    # (선택) 결과 객체의 클래스 이름 확인
    if len(result.boxes) > 0:
        print("탐지된 객체들의 클래스:")
        detected_classes = set()
        for cls_id in result.boxes.cls:
            detected_classes.add(result.names[int(cls_id)])
        print(f"- {', '.join(detected_classes)}")
    else:
        print("이미지에서 학습된 객체를 탐지하지 못했습니다.")

    # (선택) 결과 이미지를 화면에 보여주기
    result_img = cv2.imread(output_filename)
    cv2.imshow("Fine-tuned Result (Press any key to exit)", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

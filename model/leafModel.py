from ultralytics import YOLO
import cv2
import numpy as np
import os

class LeafSegmentationModel:
    def __init__(self, model_path=None):
        """
        YOLO11 세그멘테이션 모델 초기화
        
        Args:
            model_path: 학습된 모델 경로 (None이면 기본 경로 사용)
        """
        if model_path is None:
            # app/model/ 에서 app/ 로 이동
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'epoch30.pt')
        
        self.model = YOLO(model_path)
        
    def predict(self, image_path, conf=0.25, iou=0.7):
        """
        이미지에서 잎 세그멘테이션 수행
        
        Args:
            image_path: 이미지 경로
            conf: confidence 임계값
            iou: IoU 임계값
            
        Returns:
            results: YOLO 결과 객체
        """
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False
        )
        return results[0]
    
    def get_masks(self, results):
        """
        세그멘테이션 마스크 추출
        
        Returns:
            masks: numpy array (H, W) - 마스크 이미지
        """
        if results.masks is None:
            return None
        
        masks = results.masks.data.cpu().numpy()
        return masks
    
    def visualize(self, results, save_path=None):
        """
        결과 시각화
        
        Args:
            results: YOLO 결과 객체
            save_path: 저장 경로 (None이면 표시만)
        """
        annotated = results.plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated)
        
        return annotated


# 사용 예시
if __name__ == "__main__":
    # 모델 로드 (자동으로 epoch30.pt 사용)
    model = LeafSegmentationModel()
    
    # 또는 직접 경로 지정
    # model = LeafSegmentationModel("경로/epoch30.pt")
    
    # 예측 수행
    image_path = "test_image.jpg"  # 테스트 이미지 경로
    results = model.predict(image_path, conf=0.25)
    
    # 마스크 추출
    masks = model.get_masks(results)
    print(f"감지된 잎 개수: {len(masks) if masks is not None else 0}")
    
    # 결과 시각화 및 저장
    model.visualize(results, save_path="result.jpg")
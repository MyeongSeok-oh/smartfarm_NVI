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
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'epoch30.pt')
        
        self.model = YOLO(model_path)
        
        # 클래스 이름 정의
        self.class_names = {
            0: 'healthy_leaf',    # 정상 잎
            1: 'diseased_leaf'    # 이상한 잎
        }
        
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
            masks: numpy array (N, H, W) - 마스크 이미지
        """
        if results.masks is None:
            return None
        
        masks = results.masks.data.cpu().numpy()
        return masks
    
    def get_leaf_count(self, results):
        """
        정상 잎과 이상한 잎 개수 계산
        
        Args:
            results: YOLO 결과 객체
            
        Returns:
            dict: {
                'healthy': 정상 잎 개수,
                'diseased': 이상한 잎 개수,
                'total': 전체 잎 개수
            }
        """
        if results.boxes is None or len(results.boxes) == 0:
            return {
                'healthy': 0,
                'diseased': 0,
                'total': 0
            }
        
        # 클래스 ID 추출
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # 클래스별 개수 계산
        healthy_count = np.sum(class_ids == 0)
        diseased_count = np.sum(class_ids == 1)
        
        return {
            'healthy': int(healthy_count),
            'diseased': int(diseased_count),
            'total': int(healthy_count + diseased_count)
        }
    
    def get_detailed_info(self, results):
        """
        각 잎의 상세 정보 추출
        
        Returns:
            list: [
                {
                    'class': 'healthy_leaf' or 'diseased_leaf',
                    'confidence': float,
                    'bbox': [x1, y1, x2, y2],
                    'area': int (픽셀 개수)
                },
                ...
            ]
        """
        if results.boxes is None or len(results.boxes) == 0:
            return []
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # 마스크가 있으면 면적 계산
        masks = self.get_masks(results)
        
        leaf_info = []
        for i in range(len(boxes)):
            info = {
                'class': self.class_names[class_ids[i]],
                'class_id': int(class_ids[i]),
                'confidence': float(confidences[i]),
                'bbox': boxes[i].tolist()
            }
            
            # 면적 계산 (마스크가 있으면)
            if masks is not None:
                info['area'] = int(np.sum(masks[i]))
            
            leaf_info.append(info)
        
        return leaf_info
    
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
    
    def analyze_image(self, image_path, conf=0.25, iou=0.7, save_path=None):
        """
        이미지 분석 - 전체 파이프라인
        
        Args:
            image_path: 이미지 경로
            conf: confidence 임계값
            iou: IoU 임계값
            save_path: 결과 이미지 저장 경로
            
        Returns:
            dict: {
                'counts': {'healthy': int, 'diseased': int, 'total': int},
                'details': [잎 상세정보 리스트],
                'image_path': 저장된 이미지 경로
            }
        """
        # 예측
        results = self.predict(image_path, conf=conf, iou=iou)
        
        # 개수 계산
        counts = self.get_leaf_count(results)
        
        # 상세 정보
        details = self.get_detailed_info(results)
        
        # 시각화
        if save_path:
            self.visualize(results, save_path)
        
        return {
            'counts': counts,
            'details': details,
            'image_path': save_path if save_path else image_path
        }


# 사용 예시
if __name__ == "__main__":
    # 모델 로드
    model = LeafSegmentationModel()
    
    # 이미지 분석
    result = model.analyze_image(
        image_path="test_image.jpg",
        conf=0.25,
        save_path="result.jpg"
    )
    
    print("=" * 50)
    print("[잎 개수]")
    print(f"정상 잎: {result['counts']['healthy']}개")
    print(f"이상한 잎: {result['counts']['diseased']}개")
    print(f"전체: {result['counts']['total']}개")
    print("=" * 50)
    
    print("\n[상세 정보]")
    for i, leaf in enumerate(result['details'], 1):
        print(f"\n잎 {i}:")
        print(f"  - 분류: {leaf['class']}")
        print(f"  - 신뢰도: {leaf['confidence']:.2f}")
        if 'area' in leaf:
            print(f"  - 면적: {leaf['area']} pixels")
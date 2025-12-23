"""
AI 모델 추론 API
- leaf_predict: 잎 세그멘테이션 (YOLO)
- berry_predict: 딸기 탐지 (YOLO)
- series_predict: 시계열 이상 탐지 (LSTM Autoencoder)
"""

import os
import numpy as np
import pandas as pd
import cv2
from typing import Dict, Union, List

# model 폴더의 모듈 import
from model.leafModel import LeafSegmentationModel
# from model.berryModel import BerryDetectionModel  # TODO: berryModel 구현 후 추가
from model.timeModel import load_model_from_directory


class AIModel:
    """통합 AI 모델 클래스"""
    
    def __init__(self, model_dir='models'):
        """
        Args:
            model_dir: 모델 파일들이 있는 디렉토리
        """
        base_dir = os.path.dirname(__file__)
        
        # Leaf 모델 초기화
        self.leaf_model = LeafSegmentationModel()
        print("✓ Leaf 모델 로드 완료")
        
        # Berry 모델 초기화 (TODO)
        self.berry_model = None
        print("⚠ Berry 모델 미구현")
        
        # Time 모델 초기화
        time_model_dir = os.path.join(base_dir, model_dir)
        self.time_model = load_model_from_directory(time_model_dir)
        print("✓ Time 모델 로드 완료")
    
    # ============================================================
    # 1. Leaf Segmentation
    # ============================================================
    
    def leaf_predict(self, image_path: str) -> Dict:
        """
        잎 세그멘테이션 예측
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            result: 세그멘테이션 결과
        """
        # TODO(1): 잎이미지 전처리 구현하기
        if not os.path.exists(image_path):
            return {"error": f"이미지를 찾을 수 없습니다: {image_path}"}
        
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "이미지 로드 실패"}
        
        height, width, channels = image.shape
        
        # TODO(2): 센서 추론 진행
        results = self.leaf_model.predict(image_path, conf=0.25, iou=0.7)
        
        # TODO(3): 후처리 진행하기
        detections = []
        total_area = 0
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()
            
            for mask, box in zip(masks, boxes):
                x1, y1, x2, y2, conf, class_id = box
                
                # Mask 리사이즈 및 polygon 추출
                mask_resized = cv2.resize(mask, (width, height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    polygon = contour.squeeze().tolist()
                    area = cv2.contourArea(contour)
                else:
                    polygon = []
                    area = 0
                
                total_area += area
                
                detections.append({
                    "class": "leaf",
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "mask": polygon,
                    "area": int(area)
                })
        
        # TODO(4): 결과 시각화하고 저장하기
        output_path = image_path.replace('.', '_result.')
        self.leaf_model.visualize(results, save_path=output_path)
        
        # TODO(5): 모델 평가하고 반환하기
        return {
            "status": "success",
            "image_path": image_path,
            "output_path": output_path,
            "detections": detections,
            "summary": {
                "total_count": len(detections),
                "total_area": int(total_area),
                "leaf_area": int(total_area),
            },
            "image_shape": [height, width, channels],
        }
    
    # ============================================================
    # 2. Berry Detection
    # ============================================================
    
    def berry_predict(self, image_path: str) -> Dict:
        """
        딸기 탐지 예측
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            result: 탐지 결과
        """
        # TODO(1): 딸기이미지 전처리 구현하기
        if not os.path.exists(image_path):
            return {"error": f"이미지를 찾을 수 없습니다: {image_path}"}
        
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "이미지 로드 실패"}
        
        height, width, channels = image.shape
        
        # TODO(2): 센서 추론 진행
        # Berry 모델 미구현
        if self.berry_model is None:
            return {
                "status": "model_not_implemented",
                "error": "Berry 탐지 모델이 아직 구현되지 않았습니다",
                "image_path": image_path,
                "detections": [],
                "summary": {
                    "total_count": 0,
                    "ripe_count": 0,
                    "unripe_count": 0,
                },
                "image_shape": [height, width, channels],
            }
        
        # TODO(3): 후처리 진행하기
        # TODO(4): 결과 시각화하고 저장하기
        # TODO(5): 모델 평가하고 반환하기
        # TODO: berryModel 구현 후 활성화
        """
        results = self.berry_model.predict(image_path, conf=0.25)
        
        detections = []
        ripe_count = 0
        unripe_count = 0
        
        if results.boxes is not None:
            boxes = results.boxes.data.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                class_name = "ripe_berry" if int(class_id) == 0 else "unripe_berry"
                
                if class_name == "ripe_berry":
                    ripe_count += 1
                else:
                    unripe_count += 1
                
                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float(cx), float(cy)]
                })
        
        output_path = image_path.replace('.', '_berry_result.')
        self.berry_model.visualize(results, save_path=output_path)
        
        return {
            "status": "success",
            "image_path": image_path,
            "output_path": output_path,
            "detections": detections,
            "summary": {
                "total_count": len(detections),
                "ripe_count": ripe_count,
                "unripe_count": unripe_count,
            },
            "image_shape": [height, width, channels],
        }
        """
    
    # ============================================================
    # 3. Time Series Anomaly Detection
    # ============================================================
    
    def series_predict(
        self, 
        data: Union[str, pd.DataFrame, List[Dict]],
        mode: str = "batch"
    ) -> Dict:
        """
        시계열 이상 탐지 예측
        
        Args:
            data: 
                - batch 모드: CSV 파일 경로 또는 DataFrame
                - realtime 모드: 센서 데이터 버퍼 리스트
            mode: "batch" 또는 "realtime"
            
        Returns:
            result: 이상 탐지 결과
        """
        
        if mode == "batch":
            # TODO(1): 센서값 전처리 구현하기 (timeModel에서 자동 처리)
            # TODO(2): Onnx 모델 호출 해서 추론하기 (Keras 모델 사용)
            result = self.time_model.predict(data)
            
            # TODO(3): 후처리해서 값 반환하기 (True or False)
            is_anomaly = result['anomaly_count'] > 0
            status = "anomaly" if is_anomaly else "healthy"
            score = 1.0 - result['anomaly_ratio']
            
            return {
                "status": status,
                "score": float(score),
                "anomaly_detected": is_anomaly,
                "anomaly_count": result['anomaly_count'],
                "anomaly_ratio": result['anomaly_ratio'],
                "total_windows": result['total_windows'],
                "anomaly_indices": result['anomaly_indices'],
                "threshold": result['threshold'],
                "summary": result['summary'],
                "feature_importance": result['feature_importance'],
            }
        
        elif mode == "realtime":
            # TODO(1): 센서값 전처리 구현하기 (timeModel에서 자동 처리)
            # TODO(2): Onnx 모델 호출 해서 추론하기 (Keras 모델 사용)
            result = self.time_model.predict_realtime(data)
            
            if "error" in result:
                return {"error": result["error"]}
            
            # TODO(3): 후처리해서 값 반환하기 (True or False)
            status = "anomaly" if result['is_anomaly'] else "healthy"
            score = 1.0 - min(result['anomaly_score'], 1.0)
            
            return {
                "status": status,
                "score": float(score),
                "is_anomaly": result['is_anomaly'],
                "reconstruction_error": result['reconstruction_error'],
                "anomaly_score": result['anomaly_score'],
                "timestamp": result['timestamp'],
            }
        
        else:
            return {"error": f"알 수 없는 모드: {mode}"}


# ============================================================
# 편의 함수 (전역)
# ============================================================

# 전역 모델 인스턴스
_model_instance = None

def get_model() -> AIModel:
    """모델 싱글톤"""
    global _model_instance
    if _model_instance is None:
        _model_instance = AIModel()
    return _model_instance


def leaf_predict(image_path: str) -> Dict:
    """잎 세그멘테이션 (편의 함수)"""
    return get_model().leaf_predict(image_path)


def berry_predict(image_path: str) -> Dict:
    """딸기 탐지 (편의 함수)"""
    return get_model().berry_predict(image_path)


def series_predict(data, mode="batch") -> Dict:
    """시계열 이상 탐지 (편의 함수)"""
    return get_model().series_predict(data, mode)


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("AI Model Test")
    print("=" * 50)
    
    # 방법 1: 클래스로 직접 사용
    model = AIModel()
    # result = model.leaf_predict("test.jpg")
    # result = model.series_predict("data.csv")
    
    # 방법 2: 편의 함수 사용
    # result = leaf_predict("test.jpg")
    # result = series_predict("data.csv", mode="batch")
    
    print("\n테스트 완료!")
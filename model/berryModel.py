# ai_model.py - YOLO Inference Template

import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, List, Tuple

class YOLOPredictor:
    """
    YOLO Object Detection 추론 클래스
    
    C++ 버전을 Python으로 변환한 구조
    - ONNX Runtime 사용
    - 전처리: resize, BGR→RGB, normalize, HWC→NCHW
    - 후처리: NMS, 결과 dict 변환
    """
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            model_path: ONNX 모델 경로
            input_size: 모델 입력 크기 (width, height)
        """
        self.input_width, self.input_height = input_size
        
        # TODO: ONNX Runtime Session 초기화
        # - SessionOptions 설정 (intra_op, inter_op threads)
        # - Session 생성
        # - input/output names 확인
        # TODO: 메모리 미리 할당 (C++ static 변수처럼)
        # - resized image buffer
        # - float buffer
        # - input tensor buffer
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리, 이미지를 분할해서 
        
        Args:
            image: BGR 이미지 (H, W, 3)
            
        Returns:
            preprocessed: NCHW 형식 float32 텐서 (N, 3, H, W)
        """
        # TODO(1): 640*640 크기 기준으로 이미지를 N개로 크롭하기
        # TODO(2): BGR → RGB 변환 + Normalize (0~255 → 0~1)
        # TODO(3): HWC → NCHW 변환
        
        pass
    
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        모델 추론
        
        Args:
            input_tensor: 전처리된 입력 (1, 3, H, W)
            
        Returns:
            raw_output: 모델 출력 텐서
            - Detection: (1, N, 6) [x, y, w, h, conf, class]
            - Segmentation: (1, N, 6+mask_dim) 또는 별도 mask output
        """
        # TODO: ONNX Session Run
        # - input_feed 구성
        # - session.run() 실행
        # - output tensor 반환
        
        pass
    
    def postprocess_detection(self, raw_output: np.ndarray, 
                             conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detection 후처리, 위에서 나눠서 추론한걸 하나로 합쳐서 추론진행
        
        Args:
            raw_output: 모델 raw output (1, N, 6)
            conf_threshold: confidence threshold
            
        Returns:
            detections: [
                {
                    "class": class_name or class_id,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],  # xyxy 형식
                    "center": [cx, cy],
                },
                ...
            ]
        """
        # TODO(1): Confidence filtering        
        # TODO(2): NMS 적용 (cv2.dnn.NMSBoxes)
        # TODO(3): 좌표 변환 (xywh → xyxy)
        # TODO(4): 분할된 이미지 좌표계를 하나로 합치고
        # TODO(5): DICT로 반환
        
        pass
    
    def postprocess_segmentation(self, raw_output: np.ndarray,
                                 mask_output: np.ndarray,
                                 conf_threshold: float = 0.3) -> List[Dict]:
        """
        Segmentation 후처리
        
        Args:
            raw_output: detection output (1, N, 6)
            mask_output: mask output (N, H, W) or (1, N, mask_dim)
            
        Returns:
            detections: [
                {
                    "class": class_name or class_id,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],
                    "mask": [[x1,y1], [x2,y2], ...],  # polygon
                    # or "mask_binary": np.array(H, W)
                },
                ...
            ]
        """
        # TODO(1): Detection 후처리 먼저 수행
        
        # TODO(2): Mask 처리
        # - Binary mask → polygon 변환 (cv2.findContours)
        # - 또는 RLE 인코딩
        
        # TODO(3): Dict에 mask 정보 추가
        
        pass
    
    
    def visualize(self, image: np.ndarray, detections: List[Dict], 
                  save_path: str = None) -> np.ndarray:
        """
        결과 시각화
        
        Args:
            image: 원본 이미지
            detections: postprocess 결과
            save_path: 저장 경로 (optional)
            
        Returns:
            visualized: 시각화된 이미지
        """
        # TODO: bbox/mask 그리기
        # TODO: confidence, class name 표시
        # TODO: 이미지 저장 (save_path 있으면)
        
        pass


# ===== 사용 예시 =====
if __name__ == "__main__":
    # Detection
    detector = YOLOPredictor("path/to/yolo_detection.onnx")
    result = detector.predict("path/to/image.jpg")
    # result 구조는 앞서 제시한 dict 형식
    
    # Segmentation (별도 클래스 또는 flag로 구분)
    # segmentor = YOLOPredictor("path/to/yolo_seg.onnx", is_segmentation=True)
    # result = segmentor.predict("path/to/image.jpg")
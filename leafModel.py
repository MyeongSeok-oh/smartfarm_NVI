# ai_model.py - YOLO Inference Template

import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, List, Tuple

class YOLOPredictor:
    """
    YOLO Object Detection/Segmentation 추론 클래스
    
    4K 이미지 타일링 지원
    - ONNX Runtime 사용
    - 전처리: 4K → 640x640 타일 분할, BGR→RGB, normalize, HWC→NCHW
    - 후처리: NMS, 결과 dict 변환
    """
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            model_path: ONNX 모델 경로
            input_size: 모델 입력 크기 (width, height)
        """
        self.input_width, self.input_height = input_size
        self.model_path = model_path
        
        # TODO: ONNX Runtime Session 초기화
        # SessionOptions 설정
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4  # 단일 연산 내 병렬 처리
        session_options.inter_op_num_threads = 4  # 여러 연산 간 병렬 처리
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Session 생성
        self.session = ort.InferenceSession(model_path, session_options)
        
        # Input/Output names 확인
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"[INFO] 모델 로드 완료: {model_path}")
        print(f"[INFO] Input: {self.input_name}")
        print(f"[INFO] Outputs: {self.output_names}")
        
        # TODO: 메모리 미리 할당 (성능 최적화)
        # 재사용 가능한 버퍼들
        self.resized_buffer = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        self.float_buffer = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
    
    def preprocess(self, image: np.ndarray, tile_size: int = 640, overlap: int = 100) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        이미지 전처리 - 4K 이미지를 타일로 분할
        
        Args:
            image: BGR 이미지 (H, W, 3)
            tile_size: 타일 크기 (기본 640)
            overlap: 타일 간 겹침 픽셀 (기본 100)
            
        Returns:
            tiles: 전처리된 타일 리스트 [(1, 3, H, W), ...]
            positions: 각 타일의 원본 이미지 좌표 [(x, y), ...]
        """
        original_h, original_w = image.shape[:2]
        stride = tile_size - overlap
        
        tiles = []
        positions = []
        
        print(f"[INFO] 이미지 크기: {original_w}x{original_h}")
        
        # TODO(1): 640*640 크기 기준으로 이미지를 N개로 크롭하기
        for y in range(0, original_h, stride):
            for x in range(0, original_w, stride):
                # 타일 영역 계산
                x_end = min(x + tile_size, original_w)
                y_end = min(y + tile_size, original_h)
                
                # 타일 추출
                tile = image[y:y_end, x:x_end]
                
                # 타일이 640x640보다 작으면 패딩
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    tile = cv2.copyMakeBorder(
                        tile, 
                        0, tile_size - tile.shape[0],
                        0, tile_size - tile.shape[1],
                        cv2.BORDER_CONSTANT, 
                        value=[0, 0, 0]
                    )
                
                # TODO(2): BGR → RGB 변환 + Normalize (0~255 → 0~1)
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_normalized = tile_rgb.astype(np.float32) / 255.0
                
                # TODO(3): HWC → NCHW 변환
                tile_nchw = tile_normalized.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
                tile_batch = tile_nchw[np.newaxis, ...]  # (C, H, W) → (1, C, H, W)
                
                tiles.append(tile_batch)
                positions.append((x, y))
                
                # x 방향 끝에 도달하면 중단
                if x + tile_size >= original_w:
                    break
            
            # y 방향 끝에 도달하면 중단
            if y + tile_size >= original_h:
                break
        
        print(f"[INFO] 총 {len(tiles)}개 타일 생성 (overlap={overlap}px)")
        
        return tiles, positions
    
    def inference(self, input_tensor: np.ndarray) -> Tuple:
        """
        모델 추론
        
        Args:
            input_tensor: 전처리된 입력 (1, 3, H, W)
            
        Returns:
            outputs: 모델 출력 튜플
            - Detection: (boxes, scores, class_ids)
            - Segmentation: (boxes, scores, class_ids, masks)
        """
        # TODO: ONNX Session Run
        # input_feed 구성
        input_feed = {self.input_name: input_tensor.astype(np.float32)}
        
        # session.run() 실행
        outputs = self.session.run(self.output_names, input_feed)
        
        # output tensor 반환
        return outputs
    
    def postprocess_detection(self, all_outputs: List[Tuple], 
                             all_positions: List[Tuple[int, int]],
                             original_shape: Tuple[int, int],
                             conf_threshold: float = 0.5,
                             iou_threshold: float = 0.5) -> List[Dict]:
        """
        Detection 후처리 - 여러 타일의 결과를 하나로 합치기
        
        Args:
            all_outputs: 모든 타일의 추론 결과 리스트
            all_positions: 각 타일의 (x, y) 좌표
            original_shape: 원본 이미지 (H, W)
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            
        Returns:
            detections: [
                {
                    "class": class_name or class_id,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                },
                ...
            ]
        """
        original_h, original_w = original_shape
        all_detections = []
        
        # 각 타일의 결과 처리
        for outputs, (offset_x, offset_y) in zip(all_outputs, all_positions):
            # YOLO 출력 형식: (1, N, 85) - [x, y, w, h, obj_conf, class1, class2, ...]
            predictions = outputs[0][0]  # (N, 85)
            boxes = predictions[:, :4]  # xywh
            obj_scores = predictions[:, 4:5]  # objectness
            class_scores = predictions[:, 5:]  # class probabilities
            
            # 최종 신뢰도 = objectness × max(class_prob)
            scores = obj_scores * class_scores
            
            for i in range(len(predictions)):
                max_score_idx = scores[i].argmax()
                confidence = float(scores[i][max_score_idx])
                
                if confidence < conf_threshold:
                    continue
                
                # xywh → xyxy 변환 (타일 내 좌표)
                x, y, w, h = boxes[i]
                x1_tile = int(x - w/2)
                y1_tile = int(y - h/2)
                x2_tile = int(x + w/2)
                y2_tile = int(y + h/2)
                
                # 원본 이미지 좌표로 변환
                x1 = offset_x + x1_tile
                y1 = offset_y + y1_tile
                x2 = offset_x + x2_tile
                y2 = offset_y + y2_tile
                
                # 이미지 범위 체크
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                all_detections.append({
                    "class_id": int(max_score_idx),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                })
        
        print(f"[INFO] 타일 추론 완료: 총 {len(all_detections)}개 객체 탐지")
        
        # NMS로 중복 제거
        detections = self._apply_nms(all_detections, iou_threshold)
        
        print(f"[INFO] NMS 후: {len(detections)}개 객체 유지")
        
        return detections
    
    def postprocess_segmentation(self, all_outputs: List[Tuple],
                                 all_positions: List[Tuple[int, int]],
                                 original_shape: Tuple[int, int],
                                 conf_threshold: float = 0.5,
                                 iou_threshold: float = 0.5) -> List[Dict]:
        """
        Segmentation 후처리 - 여러 타일의 결과를 하나로 합치기
        
        Args:
            all_outputs: 모든 타일의 추론 결과 리스트
            all_positions: 각 타일의 (x, y) 좌표
            original_shape: 원본 이미지 (H, W)
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            
        Returns:
            detections: [
                {
                    "class": class_name or class_id,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],
                    "mask": [[x1,y1], [x2,y2], ...],  # polygon
                    "area": int
                },
                ...
            ]
        """
        original_h, original_w = original_shape
        all_detections = []
        
        # TODO(1): Detection 후처리 먼저 수행
        for outputs, (offset_x, offset_y) in zip(all_outputs, all_positions):
            # 출력 형식: boxes, scores, class_ids, masks
            boxes = outputs[0]  # (N, 4)
            scores = outputs[1]  # (N,)
            class_ids = outputs[2]  # (N,)
            masks = outputs[3]  # (N, H, W)
            
            # 신뢰도 필터링
            keep = scores > conf_threshold
            
            for i in np.where(keep)[0]:
                box = boxes[i]
                mask = masks[i]
                
                # 원본 좌표로 변환
                x1 = offset_x + int(box[0])
                y1 = offset_y + int(box[1])
                x2 = offset_x + int(box[2])
                y2 = offset_y + int(box[3])
                
                # 이미지 범위 체크
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                # TODO(2): Mask 처리
                # Binary mask → polygon 변환
                mask_resized = cv2.resize(mask, (640, 640))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_binary, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Polygon 좌표 변환
                polygon = []
                if len(contours) > 0:
                    contour = contours[0].squeeze()
                    if contour.ndim == 2:
                        for point in contour:
                            px = offset_x + int(point[0])
                            py = offset_y + int(point[1])
                            polygon.append([px, py])
                
                area = int(mask_binary.sum())
                
                # TODO(3): Dict에 mask 정보 추가
                all_detections.append({
                    "class_id": int(class_ids[i]),
                    "confidence": float(scores[i]),
                    "bbox": [x1, y1, x2, y2],
                    "mask": polygon,
                    "area": area
                })
        
        print(f"[INFO] 타일 추론 완료: 총 {len(all_detections)}개 객체 탐지")
        
        # NMS로 중복 제거
        detections = self._apply_nms(all_detections, iou_threshold)
        
        print(f"[INFO] NMS 후: {len(detections)}개 객체 유지")
        
        return detections
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Non-Maximum Suppression으로 중복 박스 제거
        
        Args:
            detections: 탐지 결과 리스트
            iou_threshold: IoU 임계값
            
        Returns:
            keep_detections: NMS 후 남은 탐지 결과
        """
        if len(detections) == 0:
            return []
        
        # confidence로 정렬
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        keep_detections = []
        
        while len(detections) > 0:
            # 가장 높은 confidence 선택
            best = detections[0]
            keep_detections.append(best)
            detections = detections[1:]
            
            # 나머지와 IoU 계산
            filtered = []
            for det in detections:
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                
                # IoU 낮으면 유지
                if iou < iou_threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep_detections
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        두 박스의 IoU 계산
        
        Args:
            box1, box2: [x1, y1, x2, y2]
            
        Returns:
            iou: Intersection over Union
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # 교집합 면적
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # 각 박스 면적
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 합집합 면적
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def visualize(self, image: np.ndarray, detections: List[Dict], 
                  class_names: Dict[int, str] = None,
                  save_path: str = None) -> np.ndarray:
        """
        결과 시각화
        
        Args:
            image: 원본 이미지
            detections: postprocess 결과
            class_names: {class_id: class_name} 매핑
            save_path: 저장 경로 (optional)
            
        Returns:
            visualized: 시각화된 이미지
        """
        # TODO: bbox/mask 그리기
        output_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_id = det["class_id"]
            
            # 클래스 이름 가져오기
            if class_names and class_id in class_names:
                label = class_names[class_id]
            else:
                label = f"class_{class_id}"
            
            # 색상 설정 (class_id에 따라)
            colors = [
                (0, 255, 0),   # 초록
                (0, 0, 255),   # 빨강
                (255, 0, 0),   # 파랑
                (255, 255, 0), # 시안
                (255, 0, 255), # 마젠타
            ]
            color = colors[class_id % len(colors)]
            
            # TODO: confidence, class name 표시
            # 박스 그리기
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # 텍스트 그리기
            text = f"{label} {confidence:.2f}"
            cv2.putText(
                output_image, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # 마스크가 있으면 그리기 (Segmentation)
            if "mask" in det and len(det["mask"]) > 0:
                polygon = np.array(det["mask"], dtype=np.int32)
                cv2.polylines(output_image, [polygon], True, color, 2)
        
        # TODO: 이미지 저장
        if save_path:
            cv2.imwrite(save_path, output_image)
            print(f"[INFO] 결과 이미지 저장: {save_path}")
        
        return output_image
    
    def predict(self, image_path: str, 
                mode: str = "detection",
                conf_threshold: float = 0.5,
                iou_threshold: float = 0.5,
                class_names: Dict[int, str] = None,
                save_path: str = None) -> Dict:
        """
        전체 추론 파이프라인
        
        Args:
            image_path: 입력 이미지 경로
            mode: "detection" 또는 "segmentation"
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            class_names: 클래스 이름 매핑
            save_path: 결과 저장 경로
            
        Returns:
            result: {
                "detections": [...],
                "image_path": str,
                "summary": {...}
            }
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")
        
        original_shape = image.shape[:2]  # (H, W)
        
        # 전처리 - 타일링
        tiles, positions = self.preprocess(image)
        
        # 각 타일 추론
        all_outputs = []
        for tile in tiles:
            outputs = self.inference(tile)
            all_outputs.append(outputs)
        
        # 후처리
        if mode == "detection":
            detections = self.postprocess_detection(
                all_outputs, positions, original_shape,
                conf_threshold, iou_threshold
            )
        else:  # segmentation
            detections = self.postprocess_segmentation(
                all_outputs, positions, original_shape,
                conf_threshold, iou_threshold
            )
        
        # 시각화
        if save_path:
            self.visualize(image, detections, class_names, save_path)
        
        # 결과 반환
        summary = {}
        if mode == "detection":
            # 클래스별 개수 집계
            class_counts = {}
            for det in detections:
                cid = det['class_id']
                class_counts[cid] = class_counts.get(cid, 0) + 1
            summary = {
                "total_count": len(detections),
                "class_counts": class_counts
            }
        else:  # segmentation
            # 면적 집계
            total_area = sum(det.get('area', 0) for det in detections)
            summary = {
                "total_count": len(detections),
                "total_area": total_area
            }
        
        return {
            "detections": detections,
            "image_path": save_path if save_path else image_path,
            "summary": summary,
            "image_shape": list(image.shape)
        }


# ===== 사용 예시 =====
if __name__ == "__main__":
    # Detection 예시
    detector = YOLOPredictor("berry_detection.onnx")
    
    result = detector.predict(
        image_path="path/to/4k_image.jpg",
        mode="detection",
        conf_threshold=0.5,
        class_names={0: "ripe_berry", 1: "unripe_berry"},
        save_path="result.jpg"
    )
    
    print(f"탐지 결과: {result['summary']}")
    
    # Segmentation 예시
    segmentor = YOLOPredictor("leaf_segmentation.onnx")
    
    result = segmentor.predict(
        image_path="path/to/4k_image.jpg",
        mode="segmentation",
        conf_threshold=0.5,
        class_names={0: "healthy_leaf", 1: "diseased_leaf"},
        save_path="seg_result.jpg"
    )
    
    print(f"세그멘테이션 결과: {result['summary']}")
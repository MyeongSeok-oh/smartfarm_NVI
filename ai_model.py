import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
from typing import Dict, List

# ============================================
# 1. 시계열 이상 탐지 (변경 없음)
# ============================================
def series_predict(temperature, humidity):
    # TODO(1): 센서값 전처리
    df = pd.DataFrame({
        'temp': temperature,
        'humidity': humidity
    })
    
    # 결측치 처리 - 선형보간
    for col in df.columns:
        df[col] = df[col].interpolate(method='linear')
        df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
    
    # 이상치 제거 - IQR 방식
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    
    # 표준화 (실제로는 학습시 mean/std 저장해서 사용)
    mean = df.mean()
    std = df.std()
    df_scaled = (df - mean) / (std + 1e-7)
    
    # 시퀀스 생성 (window_size=10 가정)
    window_size = 10
    if len(df_scaled) < window_size:
        df_scaled = pd.concat([df_scaled] * (window_size // len(df_scaled) + 1))[:window_size]
    
    X = df_scaled.values[-window_size:].reshape(1, window_size, 2).astype(np.float32)
    
    # TODO(2): ONNX 모델 추론
    ort_session = ort.InferenceSession("lstm_autoencoder.onnx")
    input_name = ort_session.get_inputs()[0].name
    reconstructed = ort_session.run(None, {input_name: X})[0]
    
    # TODO(3): 후처리 - 재구성 오차 계산
    mse = np.mean((X - reconstructed) ** 2)
    threshold = 0.1584  # 학습시 99th percentile
    
    score = max(0.0, 1 - (mse / threshold))  # 0~1 범위로 정규화
    status = "healthy" if mse < threshold else "anomaly"
    
    return {"status": status, "score": score}


# ============================================
# 2. 잎 세그멘테이션 (4K 대응)
# ============================================
def leaf_predict(image_path):
    # TODO(1): 잎 이미지 전처리
    image = cv2.imread(image_path)
    original_h, original_w = image.shape[:2]
    
    # ===== 4K 타일링 추가 =====
    input_size = 640
    overlap = 100
    stride = input_size - overlap
    
    # ONNX 모델 로드
    ort_session = ort.InferenceSession("leaf_segmentation.onnx")
    
    all_detections = []
    
    # 타일 단위로 이미지 순회
    for y in range(0, original_h, stride):
        for x in range(0, original_w, stride):
            # 타일 영역 계산
            x_end = min(x + input_size, original_w)
            y_end = min(y + input_size, original_h)
            
            # 타일 추출
            tile = image[y:y_end, x:x_end]
            
            # 타일이 640x640보다 작으면 패딩
            if tile.shape[0] < input_size or tile.shape[1] < input_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, input_size - tile.shape[0], 
                    0, input_size - tile.shape[1],
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            
            # 640x640 리사이즈 (이미 640x640이면 변화 없음)
            resized = cv2.resize(tile, (input_size, input_size))
            
            # ImageNet 정규화
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # CHW 형식 변환
            input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            
            # TODO(2): 센서 추론 진행
            outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
            
            # outputs: [boxes, scores, class_ids, masks]
            boxes, scores, class_ids, masks = outputs[0], outputs[1], outputs[2], outputs[3]
            
            # TODO(3): 후처리 진행
            confidence_threshold = 0.5
            keep = scores > confidence_threshold
            
            for i in np.where(keep)[0]:
                box = boxes[i]
                mask = masks[i]
                
                # 타일 내 좌표를 원본 이미지 좌표로 변환
                x1 = x + int(box[0] * (x_end - x) / input_size)
                y1 = y + int(box[1] * (y_end - y) / input_size)
                x2 = x + int(box[2] * (x_end - x) / input_size)
                y2 = y + int(box[3] * (y_end - y) / input_size)
                
                # 이미지 범위 체크
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                # 마스크 처리
                mask_resized = cv2.resize(mask, (input_size, input_size))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Polygon 좌표 변환
                polygon = []
                if len(contours) > 0:
                    contour = contours[0].squeeze()
                    if contour.ndim == 2:
                        for point in contour:
                            px = x + int(point[0] * (x_end - x) / input_size)
                            py = y + int(point[1] * (y_end - y) / input_size)
                            polygon.append([px, py])
                    else:
                        polygon = contour.tolist()
                
                area = int(mask_binary.sum())
                class_name = "leaf" if class_ids[i] == 0 else "ripe_berry"
                
                all_detections.append({
                    "class": class_name,
                    "confidence": float(scores[i]),
                    "bbox": [x1, y1, x2, y2],
                    "mask": polygon,
                    "area": area
                })
    
    # ===== NMS로 중복 제거 =====
    if len(all_detections) > 0:
        # confidence로 정렬
        all_detections = sorted(all_detections, key=lambda d: d['confidence'], reverse=True)
        
        keep_detections = []
        while len(all_detections) > 0:
            best = all_detections[0]
            keep_detections.append(best)
            all_detections = all_detections[1:]
            
            # IoU 계산으로 중복 제거
            filtered = []
            for det in all_detections:
                # IoU 계산
                b1, b2 = best['bbox'], det['bbox']
                x1_i = max(b1[0], b2[0])
                y1_i = max(b1[1], b2[1])
                x2_i = min(b1[2], b2[2])
                y2_i = min(b1[3], b2[3])
                
                inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                
                if iou < 0.5:  # IoU 낮으면 유지
                    filtered.append(det)
            
            all_detections = filtered
        
        detections = keep_detections
    else:
        detections = []
    
    # 면적 집계
    total_area = 0
    berry_area = 0
    leaf_area = 0
    
    for det in detections:
        total_area += det['area']
        if det['class'] == 'leaf':
            leaf_area += det['area']
        else:
            berry_area += det['area']
    
    # TODO(4): 결과 시각화하고 저장
    output_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if det["class"] == "leaf" else (0, 0, 255)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, f"{det['class']} {det['confidence']:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_path = image_path.replace('.jpg', '_result.jpg')
    cv2.imwrite(output_path, output_image)
    
    # TODO(5): 모델 평가하고 반환
    status = "healthy" if leaf_area > berry_area else "check_needed"
    
    return {
        "status": status,
        "image_path": output_path,
        "detections": detections,
        "summary": {
            "total_area": total_area,
            "berry_area": berry_area,
            "leaf_area": leaf_area,
        },
        "image_shape": list(image.shape),
    }


# ============================================
# 3. 딸기 객체 탐지 (4K 대응)
# ============================================
def berry_predict(image_path):
    # TODO(1): 딸기 이미지 전처리
    image = cv2.imread(image_path)
    original_h, original_w = image.shape[:2]
    
    # ===== 4K 타일링 추가 =====
    input_size = 640
    overlap = 100
    stride = input_size - overlap
    
    # ONNX 모델 로드
    ort_session = ort.InferenceSession("berry_detection.onnx")
    
    all_detections = []
    
    # 타일 단위로 이미지 순회
    for y in range(0, original_h, stride):
        for x in range(0, original_w, stride):
            # 타일 영역 계산
            x_end = min(x + input_size, original_w)
            y_end = min(y + input_size, original_h)
            
            # 타일 추출
            tile = image[y:y_end, x:x_end]
            
            # 타일이 640x640보다 작으면 패딩
            if tile.shape[0] < input_size or tile.shape[1] < input_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, input_size - tile.shape[0], 
                    0, input_size - tile.shape[1],
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            
            # YOLO 입력 크기
            resized = cv2.resize(tile, (input_size, input_size))
            normalized = resized.astype(np.float32) / 255.0
            input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
            
            # TODO(2): 센서 추론 진행
            outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
            
            # TODO(3): 후처리 진행
            predictions = outputs[0][0]  # (25200, 85) YOLO 출력
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            
            for i in range(len(predictions)):
                max_score_idx = scores[i].argmax()
                confidence = float(scores[i][max_score_idx])
                
                if confidence < 0.5:
                    continue
                
                # xywh → xyxy 변환 (타일 내 좌표)
                bx, by, bw, bh = boxes[i]
                x1_tile = int((bx - bw/2))
                y1_tile = int((by - bh/2))
                x2_tile = int((bx + bw/2))
                y2_tile = int((by + bh/2))
                
                # 원본 이미지 좌표로 변환
                x1 = x + int(x1_tile * (x_end - x) / input_size)
                y1 = y + int(y1_tile * (y_end - y) / input_size)
                x2 = x + int(x2_tile * (x_end - x) / input_size)
                y2 = y + int(y2_tile * (y_end - y) / input_size)
                
                # 이미지 범위 체크
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                class_name = "ripe_berry" if max_score_idx == 0 else "unripe_berry"
                
                all_detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                })
    
    # ===== NMS로 중복 제거 =====
    if len(all_detections) > 0:
        # confidence로 정렬
        all_detections = sorted(all_detections, key=lambda d: d['confidence'], reverse=True)
        
        keep_detections = []
        while len(all_detections) > 0:
            best = all_detections[0]
            keep_detections.append(best)
            all_detections = all_detections[1:]
            
            # IoU 계산으로 중복 제거
            filtered = []
            for det in all_detections:
                # IoU 계산
                b1, b2 = best['bbox'], det['bbox']
                x1_i = max(b1[0], b2[0])
                y1_i = max(b1[1], b2[1])
                x2_i = min(b1[2], b2[2])
                y2_i = min(b1[3], b2[3])
                
                inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                
                if iou < 0.5:  # IoU 낮으면 유지
                    filtered.append(det)
            
            all_detections = filtered
        
        detections = keep_detections
    else:
        detections = []
    
    # 개수 집계
    ripe_count = sum(1 for d in detections if d['class'] == 'ripe_berry')
    unripe_count = sum(1 for d in detections if d['class'] == 'unripe_berry')
    
    # TODO(4): 결과 시각화하고 저장
    output_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["class"] == "ripe_berry" else (0, 255, 0)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, f"{det['class']} {det['confidence']:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_path = image_path.replace('.jpg', '_berry_result.jpg')
    cv2.imwrite(output_path, output_image)
    
    # TODO(5): 모델 평가하고 반환
    total_count = len(detections)
    status = "healthy"
    
    return {
        "status": status,
        "image_path": output_path,
        "detections": detections,
        "summary": {
            "total_count": total_count,
            "ripe_count": ripe_count,
            "unripe_count": unripe_count,
        },
        "image_shape": list(image.shape),
    }
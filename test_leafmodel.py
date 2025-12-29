from ultralytics import YOLO
from PIL import Image
import os

def test_single_image(model_path, image_path, save_path='result.jpg'):
    """
    단일 이미지로 모델 테스트
    """
    print("=" * 60)
    print("모델 테스트 시작")
    print("=" * 60)
    
    # 모델 로드
    model = YOLO(model_path)
    print(f"✓ 모델 로드: {model_path}")
    
    # 추론
    results = model.predict(
        source=image_path,
        conf=0.01,              # 신뢰도 임계값
        save=True,              # 결과 이미지 저장
        save_txt=False,         # 라벨 저장 안 함
        show_labels=True,       # 라벨 표시
        show_conf=True,         # 신뢰도 표시
        line_width=2
    )
    
    # 결과 분석
    result = results[0]
    
    print(f"\n이미지: {image_path}")
    print(f"검출된 객체: {len(result.boxes)}개")
    
    # 클래스별 통계
    if len(result.boxes) > 0:
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        healthy_count = sum(classes == 0)
        diseased_count = sum(classes == 1)
        
        print("\n[검출 결과]")
        print(f"정상 잎: {healthy_count}개")
        print(f"병해 잎: {diseased_count}개")
        
        print("\n[신뢰도]")
        for i, (cls, conf) in enumerate(zip(classes, confs)):
            cls_name = "정상" if cls == 0 else "병해"
            print(f"  {i+1}. {cls_name}: {conf:.2%}")
    else:
        print("⚠️ 검출된 잎이 없습니다")
    
    # 저장 위치
    save_dir = result.save_dir
    print(f"\n결과 저장: {save_dir}")
    print("=" * 60)
    
    return results


# ========== 실행 ==========

# 모델 경로
model_path = r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\epoch30.pt'

# 테스트 이미지 (datasets에서 하나 선택)
test_image = r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\datasets\images\val\healthy_557015_20211021_1_0_0_3_2_12_0_410.jpg'

# 테스트 실행
results = test_single_image(model_path, test_image)
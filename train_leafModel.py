from ultralytics import YOLO
import shutil
import os

def train_leaf_segmentation_model():
    """
    정상/이상 잎 세그멘테이션 모델 학습
    """
    print("=" * 60)
    print("잎 세그멘테이션 모델 학습 시작")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(current_dir, 'data.yaml')
    
    if not os.path.exists(data_yaml):
        parent_dir = os.path.dirname(current_dir)
        data_yaml = os.path.join(parent_dir, 'data.yaml')
    
    print(f"data.yaml: {data_yaml}")
    
    if not os.path.exists(data_yaml):
        print(f"❌ data.yaml 없음")
        return None
    
    print("✓ data.yaml 확인")
    
    model = YOLO('yolo11n-seg.pt')
    
    print("\n학습 시작...")
    print(f"  - Epochs: 30")
    print(f"  - Batch: 8")
    print(f"  - Auto split: 20%")
    
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        project='runs/train',
        name='leaf_seg',
        patience=15,
        save=True,
        device=0,
        plots=True,
        verbose=True,
        split=0.2,          # 20% validation
        val=False           # ⭐⭐⭐ val 폴더 무시
    )
    
    best_model_path = model.trainer.best
    output_path = os.path.join(current_dir, 'epoch30.pt')
    
    shutil.copy(best_model_path, output_path)
    
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"모델: {output_path}")
    print("=" * 60)
    
    metrics = model.val()
    
    print(f"\n[최종 성능]")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    model_path = train_leaf_segmentation_model()
    
    if model_path:
        print(f"\n✅ 성공!")
    else:
        print(f"\n❌ 실패!")
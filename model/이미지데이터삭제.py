import shutil
from pathlib import Path
import zipfile

def move_roboflow_labels(
    roboflow_zip,
    project_datasets_folder
):
    """
    Roboflow 라벨을 프로젝트 datasets로 복사
    """
    
    # ZIP 파일 존재 확인
    zip_path = Path(roboflow_zip)
    
    if not zip_path.exists():
        print(f"❌ ZIP 파일을 찾을 수 없습니다: {roboflow_zip}")
        return
    
    print(f"ZIP 파일 확인: {zip_path.name}")
    
    # 압축 해제
    extract_dir = Path('temp_roboflow')
    
    # 기존 폴더 삭제 (있으면)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    
    print("압축 해제 중...")
    with zipfile.ZipFile(roboflow_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 압축 해제된 구조 분석
    print("\n압축 해제된 구조:")
    for item in extract_dir.rglob('*'):
        if item.is_dir():
            print(f"  📁 {item.relative_to(extract_dir)}")
    
    # train/labels 폴더 찾기
    train_labels_folders = list(extract_dir.rglob('train/labels'))
    valid_labels_folders = list(extract_dir.rglob('valid/labels'))
    
    if not train_labels_folders and not valid_labels_folders:
        print("\n❌ train/labels 또는 valid/labels 폴더를 찾을 수 없습니다!")
        print("\nZIP 파일 구조를 확인하세요.")
        shutil.rmtree(extract_dir)
        return
    
    # 변수 초기화
    train_count = 0
    val_count = 0
    
    project = Path(project_datasets_folder)
    
    # Train 라벨 복사
    if train_labels_folders:
        robo_train = train_labels_folders[0]
        project_train = project / 'labels' / 'train'
        
        print(f"\nTrain 라벨 복사 중...")
        print(f"  From: {robo_train}")
        print(f"  To: {project_train}")
        
        for lbl in robo_train.glob('*.txt'):
            shutil.copy(lbl, project_train / lbl.name)
            train_count += 1
        
        print(f"✓ {train_count}개 복사됨")
    else:
        print(f"\n⚠️ Train 라벨 없음")
    
    # Valid 라벨 복사
    if valid_labels_folders:
        robo_val = valid_labels_folders[0]
        project_val = project / 'labels' / 'val'
        
        print(f"\nVal 라벨 복사 중...")
        print(f"  From: {robo_val}")
        print(f"  To: {project_val}")
        
        for lbl in robo_val.glob('*.txt'):
            shutil.copy(lbl, project_val / lbl.name)
            val_count += 1
        
        print(f"✓ {val_count}개 복사됨")
    else:
        print(f"\n⚠️ Val 라벨 없음")
    
    # 임시 폴더 삭제
    shutil.rmtree(extract_dir)
    
    print("\n" + "="*60)
    print("✓ 라벨 복사 완료!")
    print("="*60)
    print(f"Train 라벨: {train_count}개")
    print(f"Val 라벨: {val_count}개")
    print(f"총: {train_count + val_count}개")
    print("="*60)
    
    if train_count + val_count == 0:
        print("\n⚠️ 라벨이 하나도 복사되지 않았습니다!")
        print("ZIP 파일 구조를 확인하세요.")
    else:
        print("\n✅ 다음 단계:")
        print("python train_leaf_model.py 실행")


# ========== 실행 ==========

# Roboflow ZIP 파일
roboflow_zip = r'C:\Users\mung0\Downloads\LeafSegmentation.v3i.yolov11.zip'

# 프로젝트 datasets 폴더
project_datasets = r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\datasets'

# 라벨 복사 실행
move_roboflow_labels(roboflow_zip, project_datasets)

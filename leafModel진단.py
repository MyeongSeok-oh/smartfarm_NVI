from pathlib import Path
import re

def fix_and_clean_dataset(datasets_folder):
    """
    라벨 파일명 수정 + 매칭 안 되는 이미지 삭제
    """
    datasets = Path(datasets_folder)
    
    print("="*60)
    print("데이터셋 정리 시작")
    print("="*60)
    
    for split in ['train', 'val']:
        print(f"\n[{split.upper()}]")
        
        img_dir = datasets / 'images' / split
        lbl_dir = datasets / 'labels' / split
        
        # 1. 라벨 파일명 수정
        print("1. 라벨 파일명 수정 중...")
        fixed = 0
        
        for lbl in lbl_dir.glob('*.txt'):
            old_name = lbl.name
            
            # .rf.xxx 제거
            new_name = re.sub(r'_jpg\.rf\.[a-f0-9]+\.txt$', '.txt', old_name)
            
            if new_name != old_name:
                lbl.rename(lbl_dir / new_name)
                fixed += 1
        
        print(f"   수정: {fixed}개")
        
        # 2. 라벨 목록
        label_stems = {lbl.stem for lbl in lbl_dir.glob('*.txt')}
        print(f"   라벨: {len(label_stems)}개")
        
        # 3. 매칭 안 되는 이미지 삭제
        print("2. 라벨 없는 이미지 삭제 중...")
        
        deleted = 0
        kept = 0
        
        for img in list(img_dir.glob('*.jpg')):
            if img.stem in label_stems:
                kept += 1
            else:
                img.unlink()
                deleted += 1
                
                if deleted % 500 == 0:
                    print(f"   진행: {deleted}개 삭제됨...")
        
        print(f"   유지: {kept}개")
        print(f"   삭제: {deleted}개")
        
        # 4. 최종 확인
        final_imgs = len(list(img_dir.glob('*.jpg')))
        final_lbls = len(list(lbl_dir.glob('*.txt')))
        
        print(f"3. 최종 상태:")
        print(f"   이미지: {final_imgs}개")
        print(f"   라벨: {final_lbls}개")
        
        if final_imgs == final_lbls:
            print(f"   ✓ 완벽하게 매칭됨!")
        else:
            print(f"   ⚠️ 매칭 안 됨: {abs(final_imgs - final_lbls)}개 차이")
    
    print("\n" + "="*60)
    print("✅ 데이터셋 정리 완료!")
    print("="*60)
    print("\n다음 단계:")
    print("python train_leafModel.py 다시 실행")


# ========== 실행 ==========

datasets_folder = r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\datasets'

fix_and_clean_dataset(datasets_folder)

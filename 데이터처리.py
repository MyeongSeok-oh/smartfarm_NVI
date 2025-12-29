import random
import shutil
from pathlib import Path

# Train의 일부를 Val로 복사
datasets = Path(r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\datasets')

train_imgs = list((datasets / 'images' / 'train').glob('*.jpg'))
train_lbls = list((datasets / 'labels' / 'train').glob('*.txt'))

print(f"Train: {len(train_imgs)}개")

# 20% 선택
random.seed(42)
val_count = int(len(train_imgs) * 0.2)
val_imgs = random.sample(train_imgs, val_count)

print(f"Val로 이동: {val_count}개")

# 복사 (이동 아님!)
for img in val_imgs:
    # 이미지
    shutil.copy(img, datasets / 'images' / 'val' / img.name)
    
    # 라벨
    lbl = datasets / 'labels' / 'train' / f"{img.stem}.txt"
    if lbl.exists():
        shutil.copy(lbl, datasets / 'labels' / 'val' / lbl.name)

print(f"✓ 완료!")
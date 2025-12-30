from ultralytics import YOLO
import os
import glob
import matplotlib.pyplot as plt
import random

model_path = r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\epoch30.pt'

# ⭐ 정상/병해 이미지 폴더 (실제 경로에 맞게 수정)
healthy_folder = r'C:\Users\mung0\Downloads\smartfarm_NVI-main\smartfarm_NVI-main\104.식물 병 유발 통합 데이터\01.데이터\1.Training\원천데이터\TS7_딸기_정상'
diseased_folder = r'C:\Users\mung0\Downloads\smartfarm_NVI-main\smartfarm_NVI-main\104.식물 병 유발 통합 데이터\01.데이터\1.Training\원천데이터\TS7_딸기_병해'

# 모델 로드
model = YOLO(model_path)
print("✓ 모델 로드 완료")

# 이미지 가져오기
healthy_images = glob.glob(os.path.join(healthy_folder, '*.jpg'))[:13]
diseased_images = glob.glob(os.path.join(diseased_folder, '*.jpg'))[:12]

# 섞기
test_images = healthy_images + diseased_images
random.shuffle(test_images)

print(f"✓ 정상: {len(healthy_images)}개, 병해: {len(diseased_images)}개")

# 추론
results = model.predict(
    source=test_images,
    conf=0.25,
    save=False
)

print("✓ 추론 완료")

# 5x5 격자로 출력
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for i, result in enumerate(results):
    img = result.plot()
    img = img[:, :, ::-1]  # BGR → RGB
    
    axes[i].imshow(img)
    axes[i].axis('off')
    
    # 원본 파일명 표시
    filename = os.path.basename(test_images[i])
    label = "정상" if "healthy" in filename else "병해"
    axes[i].set_title(f'{label} ({i+1})', fontsize=10)

plt.tight_layout()
plt.savefig('result_5x5_mixed.jpg', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 결과 저장: result_5x5_mixed.jpg")
os.startfile('result_5x5_mixed.jpg')
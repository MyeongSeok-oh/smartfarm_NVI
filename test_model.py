"""
실제 이상 데이터로 테스트
- 정상, 병해, 생리장애, 작물보호제처리
"""

import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from lstm_ae import CropAnomalyPipeline, load_folder_data

# 모델 로드
pipeline = CropAnomalyPipeline()
pipeline.load('models')

print("=" * 60)
print("모델 정보")
print("=" * 60)
print(f"Threshold: {pipeline.detector.threshold:.4f}")

# === 정상 데이터 (학습 안 한 20%만 테스트) ===
print("\n" + "=" * 60)
print("테스트: 정상 (학습 안 한 20%)")
print("=" * 60)

normal_df = load_folder_data('data/정상')
train_df, test_df = train_test_split(normal_df, test_size=0.2, random_state=42)

result = pipeline.predict(test_df)
print(f"데이터 수: {len(test_df)}개")
print(f"이상 비율: {result['anomaly_ratio']*100:.1f}% (낮을수록 좋음)")

# === 이상 데이터 테스트 ===
abnormal_folders = {
    '병해': 'data/병해',
    '생리장애': 'data/생리장애',
    '작물보호제처리': 'data/작물보호제처리반응'
}

for name, path in abnormal_folders.items():
    print("\n" + "=" * 60)
    print(f"테스트: {name}")
    print("=" * 60)
    
    files = glob.glob(f'{path}/*.csv')
    
    if not files:
        print(f"파일 없음: {path}")
        continue
    
    df = pd.read_csv(files[0], encoding='cp949')
    result = pipeline.predict(df)
    
    print(f"데이터 수: {len(df)}개")
    print(f"이상 비율: {result['anomaly_ratio']*100:.1f}% (높을수록 좋음)")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
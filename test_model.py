"""
실제 이상 데이터로 테스트
- 정상, 병해, 생리장애, 작물보호제처리
"""

import pandas as pd
import glob
from lstm_ae import CropAnomalyPipeline

# 모델 로드
pipeline = CropAnomalyPipeline()
pipeline.load('models')

print("=" * 60)
print("모델 정보")
print("=" * 60)
print(f"Threshold: {pipeline.detector.threshold:.4f}")

# 테스트할 폴더 목록
folders = {
    '정상': 'data/정상',
    '병해': 'data/병해',
    '생리장애': 'data/생리장애',
    '작물보호제처리': 'data/작물보호제처리반응'
}

# 각 폴더별 테스트
for name, path in folders.items():
    print("\n" + "=" * 60)
    print(f"테스트: {name}")
    print("=" * 60)
    
    files = glob.glob(f'{path}/*.csv')
    
    if not files:
        print(f"파일 없음: {path}")
        continue
    
    # 첫 번째 파일로 테스트
    df = pd.read_csv(files[0], encoding='cp949')
    result = pipeline.predict(df)
    
    print(f"파일: {files[0].split('/')[-1]}")
    print(f"데이터 수: {len(df)}개")
    print(f"이상 비율: {result['anomaly_ratio']*100:.1f}%")
    print(f"Error 범위: {result['errors'].min():.4f} ~ {result['errors'].max():.4f}")
    
    # 예상 결과 표시
    if name == '정상':
        expected = "0%에 가까워야 함"
    else:
        expected = "높을수록 좋음 (이상 탐지)"
    print(f"예상: {expected}")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
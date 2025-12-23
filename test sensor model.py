"""
센서 데이터 생성 및 모델 테스트 스크립트
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_sensor_data(
    n_samples=1000,
    anomaly_ratio=0.1,
    save_path='test_sensor_data.csv'
):
    """
    테스트용 센서 데이터 생성
    
    Args:
        n_samples: 생성할 샘플 수
        anomaly_ratio: 이상 데이터 비율 (0~1)
        save_path: CSV 저장 경로
    
    Returns:
        df: 생성된 DataFrame
    """
    print(f"센서 데이터 생성 중... (샘플: {n_samples}개)")
    
    # 시작 시간
    start_time = datetime.now() - timedelta(days=7)
    
    # 시간 생성 (10분 간격)
    timestamps = [start_time + timedelta(minutes=10*i) for i in range(n_samples)]
    
    # 정상 데이터 생성
    # 온도: 20~28도 (일중 변화 포함)
    hours = np.array([t.hour for t in timestamps])
    temp_base = 24 + 3 * np.sin(2 * np.pi * hours / 24)  # 일중 변화
    temp_noise = np.random.normal(0, 0.5, n_samples)
    temperature = temp_base + temp_noise
    
    # 습도: 50~70% (온도와 반대 경향)
    humidity_base = 60 - 5 * np.sin(2 * np.pi * hours / 24)
    humidity_noise = np.random.normal(0, 2, n_samples)
    humidity = humidity_base + humidity_noise
    
    # CO2: 400~600 ppm
    co2_base = 500 + 50 * np.sin(2 * np.pi * hours / 24 + np.pi/4)
    co2_noise = np.random.normal(0, 10, n_samples)
    co2 = co2_base + co2_noise
    
    # 이상 데이터 삽입
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    print(f"이상 데이터 {n_anomalies}개 삽입...")
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['temp_high', 'temp_low', 'humidity_high', 'co2_high'])
        
        if anomaly_type == 'temp_high':
            temperature[idx] += np.random.uniform(5, 10)
        elif anomaly_type == 'temp_low':
            temperature[idx] -= np.random.uniform(5, 10)
        elif anomaly_type == 'humidity_high':
            humidity[idx] += np.random.uniform(15, 25)
        elif anomaly_type == 'co2_high':
            co2[idx] += np.random.uniform(200, 400)
    
    # DataFrame 생성
    df = pd.DataFrame({
        '측정시각': timestamps,
        '내부 온도 1 평균': np.round(temperature, 2),
        '내부 습도 1 평균': np.round(humidity, 2),
        '내부 CO2 평균': np.round(co2, 2)
    })
    
    # CSV 저장
    df.to_csv(save_path, index=False, encoding='cp949')
    print(f"✓ 데이터 저장 완료: {save_path}")
    print(f"  - 정상: {n_samples - n_anomalies}개")
    print(f"  - 이상: {n_anomalies}개 ({anomaly_ratio*100:.1f}%)")
    
    return df


def test_time_series_model(csv_path='test_sensor_data.csv'):
    """시계열 모델 테스트"""
    print("\n" + "="*50)
    print("시계열 이상 탐지 모델 테스트")
    print("="*50)
    
    try:
        from ai_model import AIModel
        
        # 모델 로드
        print("\n모델 로딩 중...")
        model = AIModel()
        
        # Batch 예측
        print(f"\n[Batch 모드] {csv_path} 파일 예측 중...")
        result = model.series_predict(csv_path, mode="batch")
        
        print(f"\n✓ 예측 완료!")
        print(f"  - 상태: {result['status']}")
        print(f"  - 점수: {result['score']:.4f}")
        print(f"  - 이상 탐지: {result['anomaly_detected']}")
        print(f"  - 이상 개수: {result['anomaly_count']} / {result['total_windows']}")
        print(f"  - 이상 비율: {result['anomaly_ratio']*100:.2f}%")
        print(f"  - Threshold: {result['threshold']:.4f}")
        
        if result['anomaly_indices']:
            print(f"\n이상 윈도우 위치 (처음 10개):")
            print(f"  {result['anomaly_indices'][:10]}")
        
        print(f"\nFeature 중요도:")
        for feat, importance in result['feature_importance'].items():
            print(f"  {feat}: {importance:.4f}")
        
        print(f"\nSummary:")
        for key, value in result['summary'].items():
            if key != 'error_stats':
                print(f"  {key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_realtime_mode(csv_path='test_sensor_data.csv'):
    """실시간 모드 테스트"""
    print("\n" + "="*50)
    print("실시간 이상 탐지 테스트")
    print("="*50)
    
    try:
        from ai_model import AIModel
        
        # 데이터 로드
        df = pd.read_csv(csv_path, encoding='cp949')
        
        # 최근 10개 데이터로 버퍼 생성
        buffer = []
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            buffer.append({
                '측정시각': row['측정시각'],
                '내부 온도 1 평균': row['내부 온도 1 평균'],
                '내부 습도 1 평균': row['내부 습도 1 평균'],
                '내부 CO2 평균': row['내부 CO2 평균']
            })
        
        # 모델 로드
        model = AIModel()
        
        # Realtime 예측
        print(f"\n[Realtime 모드] 버퍼 {len(buffer)}개 데이터로 예측 중...")
        result = model.series_predict(buffer, mode="realtime")
        
        print(f"\n✓ 예측 완료!")
        print(f"  - 상태: {result['status']}")
        print(f"  - 점수: {result['score']:.4f}")
        print(f"  - 이상 여부: {result['is_anomaly']}")
        print(f"  - Reconstruction Error: {result['reconstruction_error']:.4f}")
        print(f"  - Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  - Timestamp: {result['timestamp']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 1. 테스트 데이터 생성
    print("1. 테스트 데이터 생성")
    print("-" * 50)
    df = generate_sensor_data(
        n_samples=500,      # 샘플 수
        anomaly_ratio=0.1,  # 10% 이상 데이터
        save_path='test_sensor_data.csv'
    )
    
    # 데이터 미리보기
    print(f"\n데이터 미리보기:")
    print(df.head())
    print(f"\n데이터 통계:")
    print(df.describe())
    
    # 2. Batch 모드 테스트
    batch_result = test_time_series_model('test_sensor_data.csv')
    
    # 3. Realtime 모드 테스트
    if batch_result:
        realtime_result = test_realtime_mode('test_sensor_data.csv')
    
    print("\n" + "="*50)
    print("테스트 완료!")
    print("="*50)
"""
LSTM Autoencoder 기반 시계열 이상 탐지
- 센서 데이터(온도, 습도, CO2)의 이상 패턴 감지
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
import joblib


class LSTMAutoencoderPredictor:
    """
    LSTM Autoencoder 기반 시계열 이상 탐지 클래스
    - 테이블 데이터(센서값 등)의 시계열 패턴 학습
    - Reconstruction Error 기반 이상 탐지
    """
    
    def __init__(self, 
                 model_path: str,
                 time_steps: int = 10,
                 n_features: int = 5,
                 scaler_params: Dict = None):
        """
        Args:
            model_path: Keras 모델 경로 (.keras)
            time_steps: 시계열 윈도우 크기 (lookback period)
            n_features: 입력 feature 개수
            scaler_params: 정규화 파라미터 {'scaler': StandardScaler, 'threshold': float}
        """
        self.time_steps = time_steps
        self.n_features = n_features
        
        # 센서 설정
        self.sensor_cols = ['내부 온도 1 평균', '내부 습도 1 평균', '내부 CO2 평균']
        self.time_col = '측정시각'
        
        # Keras 모델 로드
        self.model = load_model(model_path)
        print(f"✓ 모델 로드: {model_path}")
        
        # Scaler와 Threshold 로드
        if scaler_params:
            self.scaler = scaler_params['scaler']
            self.threshold = scaler_params['threshold']
            self.error_stats = scaler_params.get('error_stats', {})
            print(f"✓ Threshold: {self.threshold:.4f}")
        else:
            self.scaler = None
            self.threshold = None
            self.error_stats = {}
    
    def preprocess(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        테이블 데이터를 시계열 윈도우로 변환
        
        Args:
            data: 센서 데이터 (N_samples, n_features)
                 DataFrame 또는 numpy array
            
        Returns:
            windows: (N_windows, time_steps, n_features)
        """
        # (1): DataFrame → numpy 변환
        if isinstance(data, pd.DataFrame):
            data = self._preprocess_dataframe(data)
        
        # (2): Normalization (학습 시 사용한 scaler 파라미터)
        if self.scaler:
            data = self.scaler.transform(data)
        
        # (3): Sliding Window 생성
        windows = []
        for i in range(len(data) - self.time_steps + 1):
            window = data[i:i + self.time_steps]
            windows.append(window)
        windows = np.array(windows)  # (N_windows, time_steps, n_features)
        
        return windows.astype(np.float32)
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame 전처리 및 feature 추출"""
        df = df.copy()
        
        # 시간 feature 추출 (순환 인코딩)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        hour = df[self.time_col].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # 센서 데이터 전처리
        for col in self.sensor_cols:
            df[col] = df[col].replace('-', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].ffill().bfill()
            
            # IQR 이상치 처리
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        
        # Feature 선택
        feature_cols = self.sensor_cols + ['hour_sin', 'hour_cos']
        df_clean = df[feature_cols].dropna()
        
        return df_clean.values
    
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        LSTM Autoencoder 추론 (Reconstruction)
        
        Args:
            input_tensor: (N_windows, time_steps, n_features)
            
        Returns:
            reconstructed: (N_windows, time_steps, n_features)
                          원본 데이터를 재구성한 결과
        """
        # Keras 모델 예측
        reconstructed = self.model.predict(input_tensor, verbose=0)
        return reconstructed
    
    def postprocess(self, 
                    original: np.ndarray,
                    reconstructed: np.ndarray,
                    threshold: float = None) -> Dict:
        """
        Reconstruction Error 계산 및 이상 탐지
        
        Args:
            original: 원본 입력 (N_windows, time_steps, n_features)
            reconstructed: 재구성 출력 (N_windows, time_steps, n_features)
            threshold: 이상 판단 임계값 (None이면 자동 계산)
            
        Returns:
            result: {
                "reconstruction_errors": [float, ...],
                "anomaly_scores": [float, ...],
                "is_anomaly": [bool, ...],
                "threshold": float,
                "anomaly_indices": [int, ...],
            }
        """
        # (1): Reconstruction Error 계산 (MAE)
        errors = np.mean(np.abs(original - reconstructed), axis=(1, 2))
        
        # (2): Threshold 설정
        if threshold is None:
            if self.threshold is not None:
                threshold = self.threshold
            else:
                # 자동 계산: mean + 3*std
                threshold = np.mean(errors) + 3 * np.std(errors)
        
        # (3): 이상 판단
        is_anomaly = errors > threshold
        anomaly_indices = np.where(is_anomaly)[0].tolist()
        
        # (4): Anomaly Score 정규화 (0~1)
        max_error = np.max(errors)
        anomaly_scores = errors / max_error if max_error > 0 else errors
        
        return {
            "reconstruction_errors": errors.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "is_anomaly": is_anomaly.tolist(),
            "threshold": float(threshold),
            "anomaly_indices": anomaly_indices,
        }
    
    def predict(self, 
                data_path: Union[str, pd.DataFrame],
                threshold: float = None) -> Dict:
        """
        전체 추론 파이프라인
        
        Args:
            data_path: CSV 파일 경로 또는 DataFrame
            threshold: 이상 탐지 임계값
            
        Returns:
            result: 전체 예측 결과
        """
        # (1): 데이터 로드
        if isinstance(data_path, str):
            data = pd.read_csv(data_path, encoding='cp949')
        else:
            data = data_path  # DataFrame 직접 전달
        
        # (2): 전처리 (windowing)
        windows = self.preprocess(data)
        
        # (3): 추론 (reconstruction)
        reconstructed = self.inference(windows)
        
        # (4): 후처리 (anomaly detection)
        result = self.postprocess(windows, reconstructed, threshold)
        
        # (5): Feature별 중요도 계산
        feature_errors = np.mean(np.abs(windows - reconstructed), axis=(0, 1))
        feature_names = self.sensor_cols + ['hour_sin', 'hour_cos']
        feature_importance = {
            name: float(err) 
            for name, err in zip(feature_names, feature_errors)
        }
        
        # (6): Summary 생성
        summary = {
            "status": "anomaly_detected" if result["anomaly_indices"] else "normal",
            "max_error": float(np.max(result["reconstruction_errors"])),
            "avg_error": float(np.mean(result["reconstruction_errors"])),
        }
        
        if self.error_stats:
            summary["error_stats"] = self.error_stats
        dhaudtjr5948!
        
        return {
            "data_path": data_path if isinstance(data_path, str) else "dataframe",
            "total_windows": len(windows),
            "anomaly_count": len(result["anomaly_indices"]),
            "anomaly_ratio": len(result["anomaly_indices"]) / len(windows) if len(windows) > 0 else 0,
            **result,
            "feature_importance": feature_importance,
            "summary": summary,
        }
    
    def visualize(self, 
                  original_data: pd.DataFrame,
                  result: Dict,
                  save_path: str = None) -> None:
        """
        시계열 이상 탐지 결과 시각화
        
        Args:
            original_data: 원본 데이터 DataFrame
            result: predict() 결과
            save_path: 그래프 저장 경로
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. 시계열 그래프 + 이상 구간 하이라이트
        errors = result['reconstruction_errors']
        anomaly_indices = result['anomaly_indices']
        
        axes[0].plot(errors, label='Reconstruction Error', color='blue')
        axes[0].axhline(y=result['threshold'], color='red', linestyle='--', label='Threshold')
        if anomaly_indices:
            axes[0].scatter(anomaly_indices, [errors[i] for i in anomaly_indices], 
                          color='red', s=50, label='Anomaly', zorder=5)
        axes[0].set_title('Reconstruction Error Over Time')
        axes[0].set_xlabel('Window Index')
        axes[0].set_ylabel('Error')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. Anomaly Score
        anomaly_scores = result['anomaly_scores']
        axes[1].plot(anomaly_scores, color='orange', label='Anomaly Score')
        axes[1].set_title('Anomaly Score (Normalized)')
        axes[1].set_xlabel('Window Index')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. Feature 중요도
        feature_importance = result['feature_importance']
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        axes[2].bar(features, importances, color='green')
        axes[2].set_title('Feature Importance (Average Reconstruction Error)')
        axes[2].set_xlabel('Feature')
        axes[2].set_ylabel('Error')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 그래프 저장: {save_path}")
        
        plt.show()
    
    def predict_realtime(self, 
                        data_buffer: List[Dict],
                        threshold: float = None) -> Dict:
        """
        실시간 추론 (버퍼 기반)
        
        Args:
            data_buffer: 최근 time_steps개 데이터
                        [{'온도': x, '습도': y, 'CO2': z, '측정시각': t}, ...]
            threshold: 이상 판단 임계값
            
        Returns:
            realtime_result: 실시간 예측 결과
        """
        # (1): 버퍼 검증
        assert len(data_buffer) >= self.time_steps, \
            f"최소 {self.time_steps}개 데이터 필요 (현재: {len(data_buffer)}개)"
        
        # (2): DataFrame 생성 및 윈도우 생성
        df = pd.DataFrame(data_buffer[-self.time_steps:])
        window = self.preprocess(df)
        
        if len(window) == 0:
            return {"error": "전처리 실패"}
        
        # 가장 최근 윈도우 사용
        window = window[-1:]
        
        # (3): 추론
        reconstructed = self.inference(window)
        
        # (4): Error 계산 및 판단
        error = np.mean(np.abs(window - reconstructed))
        
        if threshold is None:
            threshold = self.threshold if self.threshold else error * 1.5
        
        is_anomaly = error > threshold
        
        return {
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": float(error),
            "anomaly_score": float(error / threshold) if threshold > 0 else 0.0,
            "timestamp": pd.Timestamp.now().isoformat(),
        }


def load_model_from_directory(model_dir: str = 'models') -> LSTMAutoencoderPredictor:
    """
    모델 디렉토리에서 모델과 파라미터를 로드하는 헬퍼 함수
    
    Args:
        model_dir: 모델 파일들이 있는 디렉토리
        
    Returns:
        predictor: 로드된 LSTMAutoencoderPredictor 인스턴스
    """
    model_path = Path(model_dir) / 'lstm_ae.keras'
    scaler_path = Path(model_dir) / 'scaler.pkl'
    threshold_path = Path(model_dir) / 'threshold.pkl'
    
    # Scaler 로드
    scaler = joblib.load(scaler_path)
    print(f"✓ Scaler 로드: {scaler_path}")
    
    # Threshold 로드
    threshold_data = joblib.load(threshold_path)
    
    # Predictor 생성
    predictor = LSTMAutoencoderPredictor(
        model_path=str(model_path),
        time_steps=10,
        n_features=5,
        scaler_params={
            'scaler': scaler,
            'threshold': threshold_data['threshold'],
            'error_stats': threshold_data.get('error_stats', {})
        }
    )
    
    return predictor


# ===== 사용 예시 =====
if __name__ == "__main__":
    # 방법 1: 디렉토리에서 자동 로드
    predictor = load_model_from_directory('models')
    
    # 배치 추론 (CSV 파일)
    result = predictor.predict('data/test.csv')
    
    print(f"\n총 윈도우: {result['total_windows']}")
    print(f"이상 개수: {result['anomaly_count']}")
    print(f"이상 비율: {result['anomaly_ratio']*100:.1f}%")
    print(f"상태: {result['summary']['status']}")
    
    if result['anomaly_indices']:
        print(f"\n이상 위치: {result['anomaly_indices'][:5]}...")
    
    # Feature 중요도
    print(f"\nFeature 중요도:")
    for feat, err in result['feature_importance'].items():
        print(f"  {feat}: {err:.4f}")
    
    # 시각화
    # df = pd.read_csv('data/test.csv', encoding='cp949')
    # predictor.visualize(df, result, save_path='result.png')
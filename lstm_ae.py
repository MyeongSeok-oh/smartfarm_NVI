"""
작물 상태 이상탐지 LSTM-Autoencoder
- 정상/이상 2클래스 분류
- 센서: 온도, 습도, CO2 + 시간
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# ============================================================
# 설정
# ============================================================

CONFIG = {
    'sensor_cols': ['내부 온도 1 평균', '내부 습도 1 평균', '내부 CO2 평균'],
    'time_col': '측정시각',
    'timesteps': 10,
    'n_features': 5,  # 온도, 습도, CO2, hour_sin, hour_cos
    'latent_dim': 16,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 30,
    'batch_size': 64,
    'threshold_sigma': 3,
}


# ============================================================
# 데이터 로드
# ============================================================

def load_folder_data(folder_path):
    """폴더 내 모든 CSV 파일을 하나의 DataFrame으로 합침"""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {folder_path}")
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(file, encoding='cp949')
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"로드 완료: {len(all_files)}개 파일, {len(combined_df)}개 행")
    
    return combined_df


# ============================================================
# 데이터 전처리
# ============================================================

class DataPreprocessor:
    def __init__(self, sensor_cols=CONFIG['sensor_cols'], time_col=CONFIG['time_col']):
        self.sensor_cols = sensor_cols
        self.time_col = time_col
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _extract_time_features(self, df):
        """시간 정보 추출 (순환 인코딩)"""
        df = df.copy()
        
        # 측정시각을 datetime으로 변환
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # hour 추출 (0~23)
        hour = df[self.time_col].dt.hour
        
        # 순환 인코딩 (24시간 주기)
        # sin/cos로 변환하면 23시와 0시가 가깝게 표현됨
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        return df
    
    def clean_data(self, df):
        df = df.copy()
        
        # 시간 feature 추출
        df = self._extract_time_features(df)
        
        # 센서 컬럼 처리
        for col in self.sensor_cols:
            df[col] = df[col].replace('-', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].ffill().bfill()
            
            # IQR 이상치 처리
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        
        # 최종 컬럼 선택 (센서 + 시간)
        feature_cols = self.sensor_cols + ['hour_sin', 'hour_cos']
        df = df[feature_cols].dropna()
        
        return df
    
    def fit_transform(self, df):
        df = self.clean_data(df)
        self.is_fitted = True
        return self.scaler.fit_transform(df.values)
    
    def transform(self, df):
        df = self.clean_data(df)
        return self.scaler.transform(df.values)
    
    def save(self, path):
        joblib.dump(self.scaler, path)
    
    def load(self, path):
        self.scaler = joblib.load(path)
        self.is_fitted = True


def create_sequences(data, timesteps=CONFIG['timesteps']):
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)


# ============================================================
# LSTM-Autoencoder 모델
# ============================================================

def build_lstm_autoencoder():
    timesteps = CONFIG['timesteps']
    n_features = CONFIG['n_features']
    latent_dim = CONFIG['latent_dim']
    dropout = CONFIG['dropout_rate']
    
    inputs = Input(shape=(timesteps, n_features))
    
    # === Encoder (2층) ===
    x = LSTM(32, activation='tanh', return_sequences=True)(inputs)  # 추가
    x = LSTM(latent_dim, activation='tanh', return_sequences=False)(x)
    x = Dropout(dropout)(x)
    
    # === Decoder (2층) ===
    x = RepeatVector(timesteps)(x)
    x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    x = LSTM(32, activation='tanh', return_sequences=True)(x)  # 추가
    x = Dropout(dropout)(x)
    outputs = TimeDistributed(Dense(n_features))(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), loss='mae')
    
    return model


# ============================================================
# 이상탐지 클래스
# ============================================================

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.threshold = None
        self.error_stats = {}
    
    def fit(self, X_normal, verbose=1):
        print(f"학습 데이터: {X_normal.shape}")
        
        self.model = build_lstm_autoencoder()
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        
        self.model.fit(
            X_normal, X_normal,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_split=0.1,
            callbacks=callbacks,
            verbose=verbose
        )
        
        errors = self._calculate_error(X_normal)
        mean, std = np.mean(errors), np.std(errors)
        self.threshold = mean + CONFIG['threshold_sigma'] * std
        
        self.error_stats = {'mean': mean, 'std': std, 'min': errors.min(), 'max': errors.max()}
        
        print(f"\nThreshold: {self.threshold:.4f} (mean={mean:.4f}, std={std:.4f})")
    
    def _calculate_error(self, X):
        reconstructed = self.model.predict(X, verbose=0)
        return np.mean(np.abs(X - reconstructed), axis=(1, 2))
    
    def predict(self, X):
        errors = self._calculate_error(X)
        predictions = (errors > self.threshold).astype(int)
        return predictions, errors
    
    def evaluate(self, X_test, y_true):
        predictions, errors = self.predict(X_test)
        
        print("\n[Confusion Matrix]")
        cm = confusion_matrix(y_true, predictions)
        print(f"  정상→정상: {cm[0,0]}, 정상→이상: {cm[0,1]}")
        print(f"  이상→정상: {cm[1,0]}, 이상→이상: {cm[1,1]}")
        
        print("\n[Classification Report]")
        print(classification_report(y_true, predictions, target_names=['정상', '이상']))
        
        return predictions, errors
    
    def save(self, model_path, threshold_path):
        self.model.save(model_path)
        joblib.dump({'threshold': self.threshold, 'error_stats': self.error_stats}, threshold_path)
    
    def load(self, model_path, threshold_path):
        self.model = load_model(model_path)
        data = joblib.load(threshold_path)
        self.threshold = data['threshold']
        self.error_stats = data['error_stats']


# ============================================================
# 파이프라인
# ============================================================

class CropAnomalyPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.detector = AnomalyDetector()
    
    def train(self, normal_df, verbose=1):
        normalized = self.preprocessor.fit_transform(normal_df)
        sequences = create_sequences(normalized)
        self.detector.fit(sequences, verbose=verbose)
    
    def predict(self, df):
        normalized = self.preprocessor.transform(df)
        sequences = create_sequences(normalized)
        predictions, errors = self.detector.predict(sequences)
        
        return {
            'predictions': predictions,
            'errors': errors,
            'anomaly_ratio': np.mean(predictions),
            'threshold': self.detector.threshold
        }
    
    def evaluate(self, test_df, labels):
        normalized = self.preprocessor.transform(test_df)
        sequences = create_sequences(normalized)
        adjusted_labels = labels[CONFIG['timesteps'] - 1:]
        return self.detector.evaluate(sequences, adjusted_labels)
    
    def save(self, directory='models'):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.preprocessor.save(f'{directory}/scaler.pkl')
        self.detector.save(f'{directory}/lstm_ae.keras', f'{directory}/threshold.pkl')
        print(f"저장 완료: {directory}/")
    
    def load(self, directory='models'):
        self.preprocessor.load(f'{directory}/scaler.pkl')
        self.detector.load(f'{directory}/lstm_ae.keras', f'{directory}/threshold.pkl')
        print(f"로드 완료: {directory}/")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("정상 데이터 로드")
    print("=" * 50)
    normal_df = load_folder_data('data/정상')
    
    print("\n" + "=" * 50)
    print("모델 학습 (시간 정보 포함)")
    print("=" * 50)
    print(f"Features: 온도, 습도, CO2, hour_sin, hour_cos")
    
    pipeline = CropAnomalyPipeline()
    pipeline.train(normal_df)
    
    pipeline.save('models')
    
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
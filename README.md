# 작물 이상탐지 LSTM-Autoencoder

LSTM-Autoencoder 기반 작물 환경 이상탐지 시스템

## 개요

센서 데이터(온도, 습도, CO2, 시간)를 활용하여 작물의 정상/이상 상태를 판별하는 딥러닝 모델입니다.

- **모델**: LSTM-Autoencoder (2층)
- **입력**: 온도, 습도, CO2, 시간(hour_sin, hour_cos)
- **출력**: 정상(0) / 이상(1)

---

## 데이터 전처리

### 1. 결측치 처리

센서 오류나 통신 끊김으로 발생한 빈 값을 처리합니다.

```python
# '-' 값을 NaN으로 변환
df[col] = df[col].replace('-', np.nan)

# 선형 보간 (앞뒤 값의 평균으로 채움)
df[col] = df[col].interpolate(method='linear')

# 맨 앞/뒤는 인접값으로 채움
df[col] = df[col].ffill().bfill()
```

**예시**:
```
원본: [20, '-', 22, NaN, 24]
결과: [20, 21, 22, 23, 24]
```

### 2. 이상치 처리 (IQR 방식)

극단적인 센서 오류값을 제거합니다.

```python
Q1, Q3 = df[col].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR  # 하한
upper = Q3 + 1.5 * IQR  # 상한
df[col] = df[col].clip(lower, upper)
```

**예시**:
```
원본: [15, 18, 20, 22, 25, 100]
         Q1=18, Q3=25, IQR=7
         하한=7.5, 상한=35.5
결과: [15, 18, 20, 22, 25, 35.5]  ← 100이 35.5로 제한됨
```

### 3. 시간 정보 추출 (순환 인코딩)

시간을 sin/cos로 변환하여 23시와 0시가 가깝게 표현되도록 합니다.

```python
hour = df['측정시각'].dt.hour  # 0~23

df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
```

**왜 sin/cos를 사용하는가?**
```
일반 숫자: 0시와 23시의 거리 = 23 (멀다고 인식)
sin/cos:   0시와 23시의 거리 ≈ 0.26 (가깝다고 인식) ✓

시계처럼 원형으로 표현해서 자정 전후가 가깝게 됨
```

**변환 결과**:
| 시간 | hour_sin | hour_cos |
|------|----------|----------|
| 0시 | 0.00 | 1.00 |
| 6시 | 1.00 | 0.00 |
| 12시 | 0.00 | -1.00 |
| 18시 | -1.00 | 0.00 |
| 23시 | -0.26 | 0.97 |

### 4. 정규화 (StandardScaler)

모든 값을 평균=0, 표준편차=1로 변환합니다.

```python
정규화 값 = (원본 값 - 평균) / 표준편차
```

**예시**:
```
온도: [15, 20, 25] → 평균=20, 표준편차=5
결과: [-1, 0, 1]
```

### 5. 시퀀스 생성

연속된 10개 데이터를 하나의 시퀀스로 묶습니다.

```python
def create_sequences(data, timesteps=10):
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)
```

**예시**:
```
원본: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

시퀀스1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
시퀀스2: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
시퀀스3: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### 전처리 흐름 요약

```
원본 CSV 데이터
       ↓
① 결측치 처리 (-, NaN → 선형 보간)
       ↓
② 이상치 처리 (IQR 범위 밖 → 경계값)
       ↓
③ 시간 추출 (hour → sin/cos)
       ↓
④ 정규화 (평균=0, 표준편차=1)
       ↓
⑤ 시퀀스 생성 (10개씩 묶음)
       ↓
모델 입력 데이터 (samples, 10, 5)
```

---

## 모델 구조

### LSTM-Autoencoder 아키텍처 (2층)

```
┌─────────────────────────────────────┐
│            Input (10, 5)            │  ← 10개 시점 × 5개 변수
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  LSTM Encoder Layer 1 (32 units)    │  ← 1층: 시퀀스 유지
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  LSTM Encoder Layer 2 (16 units)    │  ← 2층: 압축
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│           Dropout (0.2)             │  ← 과적합 방지
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      Latent Space (16 dim)          │  ← 핵심 패턴만 저장
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│        RepeatVector (10)            │  ← 10개 시점으로 복제
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  LSTM Decoder Layer 1 (16 units)    │  ← 1층: 복원 시작
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  LSTM Decoder Layer 2 (32 units)    │  ← 2층: 확장
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│           Dropout (0.2)             │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│   TimeDistributed Dense (5)         │  ← 각 시점별 5개 값 출력
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│           Output (10, 5)            │  ← 복원된 데이터
└─────────────────────────────────────┘
```

### 모델 코드

```python
def build_lstm_autoencoder():
    timesteps = CONFIG['timesteps']
    n_features = CONFIG['n_features']
    latent_dim = CONFIG['latent_dim']
    dropout = CONFIG['dropout_rate']
    
    inputs = Input(shape=(timesteps, n_features))
    
    # === Encoder (2층) ===
    x = LSTM(32, activation='tanh', return_sequences=True)(inputs)
    x = LSTM(latent_dim, activation='tanh', return_sequences=False)(x)
    x = Dropout(dropout)(x)
    
    # === Decoder (2층) ===
    x = RepeatVector(timesteps)(x)
    x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    x = LSTM(32, activation='tanh', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    outputs = TimeDistributed(Dense(n_features))(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), loss='mae')
    
    return model
```

### 레이어별 return_sequences 설정

| 레이어 | return_sequences | 설명 |
|--------|------------------|------|
| Encoder Layer 1 | True | 다음 LSTM으로 시퀀스 전달 |
| Encoder Layer 2 | False | 압축 (마지막 시점만 출력) |
| Decoder Layer 1 | True | 시퀀스 유지 |
| Decoder Layer 2 | True | 시퀀스 유지 |

### 모델 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| timesteps | 10 | 시퀀스 길이 |
| n_features | 5 | 입력 변수 (온도, 습도, CO2, hour_sin, hour_cos) |
| latent_dim | 16 | 압축 차원 |
| dropout_rate | 0.2 | 드롭아웃 비율 |
| learning_rate | 0.001 | 학습률 |
| epochs | 30 | 최대 학습 반복 횟수 |
| batch_size | 64 | 배치 크기 |
| loss | MAE | 손실 함수 (Mean Absolute Error) |

### 이상 판정 방식

```
Reconstruction Error = |원본 - 복원| 의 평균

if Error > Threshold:
    이상 (1)
else:
    정상 (0)

Threshold = mean + 3 × std (정상 데이터 기준)
```

**동작 원리**:
```
정상 데이터 → 모델이 잘 복원 → Error 낮음 (예: 0.1)
이상 데이터 → 모델이 복원 실패 → Error 높음 (예: 2.5)
```

---

## 파이프라인

### 전체 구조

```python
class CropAnomalyPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()  # 전처리기
        self.detector = AnomalyDetector()       # 이상탐지기
```

### 학습 파이프라인

```
정상 CSV 파일들
       ↓
┌──────────────────────────────────┐
│     load_folder_data()           │  폴더 내 CSV 합치기
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  preprocessor.fit_transform()    │  전처리 + 정규화 학습
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│      create_sequences()          │  시퀀스 생성
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│       detector.fit()             │  LSTM-AE 학습
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│    Threshold 계산 및 저장        │  mean + 3×std
└──────────────────────────────────┘
```

### 예측 파이프라인

```
새로운 CSV 데이터
       ↓
┌──────────────────────────────────┐
│   preprocessor.transform()       │  전처리 (학습된 스케일러 사용)
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│      create_sequences()          │  시퀀스 생성
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│      detector.predict()          │  복원 → 에러 계산
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│    Error > Threshold 비교        │
└──────────────┬───────────────────┘
               ↓
        정상(0) / 이상(1)
```

### 사용 예시

```python
from lstm_ae import CropAnomalyPipeline, load_folder_data

# === 학습 ===
pipeline = CropAnomalyPipeline()
normal_df = load_folder_data('data/정상')
pipeline.train(normal_df)
pipeline.save('models')

# === 예측 ===
pipeline.load('models')
result = pipeline.predict(new_df)

print(result['predictions'])    # [0, 0, 1, 1, ...] 
print(result['anomaly_ratio'])  # 이상 비율 (0.0 ~ 1.0)
print(result['errors'])         # 각 시퀀스별 에러값
print(result['threshold'])      # 판정 기준값
```

---

## 설치

```bash
# 저장소 클론
git clone https://github.com/kennychae/agrigotchi.git
cd agrigotchi

# 패키지 설치
pip install -r requirements.txt
```

### 필수 패키지

| 패키지 | 용도 |
|--------|------|
| tensorflow | 딥러닝 모델 |
| pandas | 데이터 처리 |
| numpy | 수치 계산 |
| scikit-learn | 전처리, 평가 |
| joblib | 모델 저장/로드 |

### Windows 추가 설치

TensorFlow 실행 시 DLL 오류 발생 시:
- [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 설치

---

## 데이터 구조

```
data/
├── 정상/
│   └── *.csv
├── 병해/
│   └── *.csv
├── 생리장애/
│   └── *.csv
└── 작물보호제처리반응/
    └── *.csv
```

### CSV 필수 컬럼

| 컬럼명 | 설명 |
|--------|------|
| 측정시각 | YYYY-MM-DD HH:MM:SS |
| 내부 온도 1 평균 | 온도 (°C) |
| 내부 습도 1 평균 | 습도 (%) |
| 내부 CO2 평균 | CO2 농도 (ppm) |

---

## 성능 개선 가이드

| 순서 | 방법 | 난이도 | 효과 |
|------|------|--------|------|
| 1 | 데이터 확보 | ★☆☆ | ★★★ |
| 2 | threshold_sigma 조절 (2~4) | ★☆☆ | ★★☆ |
| 3 | epochs 증가 (50, 100) | ★☆☆ | ★★☆ |
| 4 | feature 추가 (이슬점 등) | ★★☆ | ★★☆ |
| 5 | 모델 구조 변경 (latent_dim, timesteps) | ★★★ | ★★☆ |

---

## 파일 구조

```
agrigotchi/
├── lstm_ae_model/
│   ├── lstm_ae.py          # 메인 모델
│   ├── test_model.py       # 테스트
│   ├── requirements.txt
│   └── README.md
├── data/
└── models/
    ├── lstm_ae.keras
    ├── scaler.pkl
    └── threshold.pkl
```

## 참고 논문

Wei, Y., et al. (2022). "LSTM-Autoencoder based Anomaly Detection for Indoor Air Quality Time Series Data"

## 라이선스

MIT License
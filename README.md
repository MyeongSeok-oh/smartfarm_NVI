# 🌱 Agrigotchi - AI 스마트 농업 모니터링 시스템

작물의 건강 상태를 실시간으로 모니터링하고 이상 징후를 자동으로 감지하는 AI 기반 스마트 농업 시스템

## 📋 프로젝트 개요

Agrigotchi는 컴퓨터 비전과 시계열 분석을 결합하여 온실 내 작물을 모니터링합니다:
- **이미지 분석**: 잎 세그멘테이션 및 열매 감지 (YOLOv8)
- **센서 데이터 분석**: 환경 센서 데이터를 통한 이상 감지 (LSTM-Autoencoder)

## ✨ 주요 기능

### 1. 컴퓨터 비전 모듈
- **잎 세그멘테이션**: 작물의 잎 영역 정확히 분할
- **열매 감지**: 성숙도 및 병해 확인을 위한 열매 탐지
- **고해상도 처리**: 4K 이미지를 640x640 타일로 분할하여 처리 (100px 중첩)

### 2. 시계열 이상 감지
- **센서 모니터링**: 온도, 습도, CO2 등 환경 데이터 분석
- **이상 탐지**: 정상 데이터로 학습한 모델이 비정상 패턴 자동 감지
- **정확도**: 94% 정확도, 99.86% 정밀도

## 🏗️ 시스템 아키텍처

```
app/model/
├── ai_model.py        # 메인 통합 모듈
├── leafModel.py       # 잎 세그멘테이션 모델
├── berryModel.py      # 열매 감지 모델
└── timeModel.py       # 시계열 이상 감지 모델
```

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/agrigotchi.git
cd agrigotchi
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 모델 파일 다운로드
학습된 모델 파일을 다운로드하여 `app/model/` 디렉토리에 저장:

📦 [모델 다운로드 (Google Drive)](https://drive.google.com/file/d/1cLAqmtVaBiOB_9MxvLT11UZolU20zmAl/view?usp=sharing)

```
app/model/
├── leaf_model.keras
├── berry_model.pt
└── time_model.keras
```

## 💻 사용 방법

### 기본 실행
```python
from app.model.ai_model import AIModel

# 모델 초기화
model = AIModel()

# 이미지 분석
leaf_result = model.analyze_leaf(image_path)
berry_result = model.detect_berry(image_path)

# 센서 데이터 이상 감지
anomaly_result = model.detect_anomaly(sensor_data)
```

### 센서 데이터 전처리
```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('sensor_data.csv')

# 전처리 (결측치 + 이상치)
for col in ['temp', 'humidity', 'co2']:
    # 선형보간으로 결측치 처리
    df[col] = df[col].interpolate(method='linear')
    
    # IQR 방식으로 이상치 처리
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```

## 📊 모델 성능

| 모델 | 작업 | 정확도/성능 |
|------|------|-------------|
| LSTM-Autoencoder | 시계열 이상 감지 | 94% 정확도, 99.86% 정밀도 |
| YOLOv8 | 잎 세그멘테이션 | 고해상도 이미지 지원 |
| YOLOv8 | 열매 감지 | 실시간 처리 가능 |

## 🛠️ 기술 스택

- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Computer Vision**: YOLOv8, OpenCV
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Time Series**: LSTM-Autoencoder
- **Deployment**: Raspberry Pi 호환 (모델 크기 ~700KB, 메모리 ~150MB)

## 📁 데이터 전처리

센서 데이터 전처리 세부 방법은 [`docs/데이터_전처리_가이드.md`](docs/데이터_전처리_가이드.md) 참조

**핵심 전처리 파이프라인**:
1. 결측치 처리: 선형보간 (interpolate)
2. 이상치 제거: IQR 방식
3. 정규화: StandardScaler

## 📦 Requirements

```
tensorflow>=2.13.0
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## 👥 팀 구성

- AI 모델 통합 및 시스템 아키텍처
- 모델 학습 및 최적화
- 파일 경로 관리 및 데이터 파이프라인

## 📄 라이선스

This project is licensed under the MIT License.

## 📞 문의

프로젝트 관련 문의사항은 이슈로 등록해주세요.
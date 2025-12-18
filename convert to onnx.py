"""
Keras 모델(.h5)을 ONNX로 변환
"""

import tensorflow as tf
import tf2onnx
import onnx
import os

print("=" * 60)
print("Keras → ONNX 변환")
print("=" * 60)

# ===== 1. Keras 모델 로드 =====
print("\n[1단계] Keras 모델 로드")

model_path = "lstm_ae_model.h5"  # 실제 모델 경로로 수정

if not os.path.exists(model_path):
    print(f"❌ 모델 파일이 없습니다: {model_path}")
    print("\n저장된 모델 위치를 확인하세요:")
    print("  - models/lstm_ae_model.h5")
    print("  - /mnt/user-data/outputs/lstm_ae_model.h5")
    exit()

model = tf.keras.models.load_model(model_path)
print(f"✅ 모델 로드 완료: {model_path}")
print(f"   입력 shape: {model.input_shape}")
print(f"   출력 shape: {model.output_shape}")

# ===== 2. ONNX로 변환 =====
print("\n[2단계] ONNX 변환")

# 입력 스펙 정의
input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]

# 변환
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13  # ONNX opset 버전
)

print("✅ ONNX 변환 완료")

# ===== 3. ONNX 모델 저장 =====
print("\n[3단계] ONNX 모델 저장")

onnx_path = "/mnt/user-data/outputs/lstm_ae_model.onnx"
onnx.save(onnx_model, onnx_path)

# 파일 크기 확인
file_size = os.path.getsize(onnx_path) / 1024  # KB
print(f"✅ 저장 완료: {onnx_path}")
print(f"   파일 크기: {file_size:.1f} KB")

# ===== 4. ONNX 모델 검증 =====
print("\n[4단계] ONNX 모델 검증")

try:
    # ONNX 모델 로드 및 검증
    onnx_model_check = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model_check)
    print("✅ ONNX 모델 유효성 검증 완료")
    
    # 모델 정보 출력
    print(f"\n[모델 정보]")
    print(f"  IR 버전: {onnx_model_check.ir_version}")
    print(f"  Producer: {onnx_model_check.producer_name}")
    print(f"  입력:")
    for input in onnx_model_check.graph.input:
        print(f"    - {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    print(f"  출력:")
    for output in onnx_model_check.graph.output:
        print(f"    - {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
        
except Exception as e:
    print(f"❌ 검증 실패: {e}")

# ===== 5. 추론 테스트 (선택) =====
print("\n[5단계] 추론 테스트")

try:
    import onnxruntime as ort
    import numpy as np
    
    # ONNX Runtime 세션 생성
    sess = ort.InferenceSession(onnx_path)
    
    # 테스트 입력 생성
    test_input = np.random.randn(1, 10, 3).astype(np.float32)
    
    # 추론
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    result = sess.run([output_name], {input_name: test_input})
    
    print(f"✅ 추론 테스트 성공")
    print(f"   입력 shape: {test_input.shape}")
    print(f"   출력 shape: {result[0].shape}")
    
except ImportError:
    print("⚠️ onnxruntime이 설치되지 않아 추론 테스트 생략")
    print("   설치: pip install onnxruntime")
except Exception as e:
    print(f"❌ 추론 테스트 실패: {e}")

# ===== 요약 =====
print("\n" + "=" * 60)
print("변환 완료 요약")
print("=" * 60)
print(f"""
원본 모델: {model_path}
ONNX 모델: {onnx_path}
파일 크기: {file_size:.1f} KB

다음 단계:
1. Raspberry Pi에 onnxruntime 설치
   pip install onnxruntime

2. ONNX 모델 로드 및 추론
   import onnxruntime as ort
   sess = ort.InferenceSession('lstm_ae_model.onnx')
   result = sess.run([output_name], {{input_name: input_data}})
""")

print("완료!")
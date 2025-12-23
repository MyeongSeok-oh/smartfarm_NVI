import torch

model_path = r'C:\Users\mung0\OneDrive\바탕 화면\smartfarm\agrigotchi-main\app\epoch30.pt'

checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

print("=== 기본 정보 ===")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Fitness: {checkpoint['best_fitness']}")
print(f"Version: {checkpoint.get('version', 'N/A')}")

print("\n=== 모델 구조 ===")
model = checkpoint['model']
print(type(model))
print(model)

print("\n=== 학습 인자 ===")
print(checkpoint.get('train_args', 'N/A'))
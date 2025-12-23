#sensor_routes.py 센서 -> 서버 API
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class SensorData(BaseModel):
    device_id: str
    temperature: float
    humidity: float
    image: str  # base64

@router.post("/api/sensor/data")
async def receive_sensor_data(data: SensorData):
    timestamp = datetime.now()
    print(f"[{timestamp}] Sensor {data.device_id} → {data.temperature}°C, {data.humidity}%")

    # TODO: 저장(DB)
    # TODO: 이미지 저장
    # TODO: AI 모델 호출

    return {"status": "ok"}
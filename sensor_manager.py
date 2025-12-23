#sensor_manager.py 센서 검색 및 통신

from app.database import SessionLocal, SensorData

def save_sensor_data(temp, hum, co2, ai_result):
    db = SessionLocal()
    data = SensorData(
        temperature=temp,
        humidity=hum,
        co2=co2,
        ai_result=ai_result
    )
    db.add(data)
    db.commit()
    db.close()
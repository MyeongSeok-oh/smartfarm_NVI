#database.py 데이터베이스 관련

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from zoneinfo import ZoneInfo
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sensor.db")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul"))

class SensorData(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=now_kst())
    temperature = Column(Float)
    humidity = Column(Float)
    co2 = Column(Float)
    ai_result = Column(String)  # JSON string or text

def init_db():
    Base.metadata.create_all(bind=engine)
#setup_routes.py 초기 설정
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.config_manager import load_config, save_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request):
    return templates.TemplateResponse("setup.html", {"request": request})

@router.post("/setup_complete")
def setup_complete():
    cfg = load_config()
    cfg["setup_completed"] = True
    save_config(cfg)
    return {"status": "ok"}

@router.post("/change_password")
def change_password(new_password: str = Form(...)):
    cfg = load_config()
    cfg["admin_password"] = new_password
    save_config(cfg)
    return {"status": "ok", "message": "비밀번호가 변경되었습니다!"}

@router.post("/register_sensor")
def register_sensor(device_id: str = Form(...)):
    cfg = load_config()
    cfg["sensors"].append({"id": device_id})
    save_config(cfg)
    return {"status": "ok", "message": "센서가 등록되었습니다!"}
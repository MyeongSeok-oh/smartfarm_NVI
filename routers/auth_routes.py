#auth_routes.py 로그인 라우터
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.config_manager import load_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    cfg = load_config()

    if username == cfg["admin_user"] and password == cfg["admin_password"]:

        # 첫 설정이 끝나지 않은 경우 → setup 이동
        if not cfg.get("setup_completed", False):
            response = RedirectResponse("/setup", status_code=302)
        else:
            response = RedirectResponse("/dashboard", status_code=302)

        response.set_cookie("logged_in", "yes")
        return response

    else:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "로그인 실패"}
        )

@router.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("logged_in")
    return response
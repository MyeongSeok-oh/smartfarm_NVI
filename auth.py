#auth.py 로그인 보안 관련
from fastapi import Request

def is_logged_in(request: Request):
    return request.cookies.get("logged_in") == "yes"
#config_manager.py config 저장 및 로드
import json
import os

CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "admin_user": "admin",
    "admin_password": "1234",
    "setup_completed": False,
    "sensors": []
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
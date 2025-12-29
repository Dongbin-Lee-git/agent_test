import logging
import os
from datetime import datetime

# 로그 디렉토리 생성
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 파일 경로 설정
log_filename = f"agent_flow_{datetime.now().strftime('%Y%m%d')}.log"
log_path = os.path.join(LOG_DIR, log_filename)

# 로거 설정
logger = logging.getLogger("agent_flow")
logger.setLevel(logging.INFO)

# 기존 핸들러 제거 (중복 방지)
if logger.hasHandlers():
    logger.handlers.clear()

# 파일 핸들러 설정
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 설정 (선택 사항)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

from typing import Any

def log_agent_step(agent_name: str, step_description: str, data: Any = None):
    message = f"[{agent_name}] {step_description}"
    if data:
        message += f"\nData: {data}"
    logger.info(message)
    logger.info("-" * 50)

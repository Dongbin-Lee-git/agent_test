#!/bin/bash

# 백엔드 실행 (백그라운드)
echo "Starting Backend (Uvicorn)..."
uvicorn main:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# 프론트엔드 실행 (백그라운드)
echo "Starting Frontend (Streamlit)..."
streamlit run infra/frontend/ui.py --server.port 8002 --server.address 0.0.0.0 &
FRONTEND_PID=$!

# 하나라도 종료되면 전체 종료되도록 감시
wait -n

# 종료 시 자식 프로세스 정리
kill $BACKEND_PID $FRONTEND_PID

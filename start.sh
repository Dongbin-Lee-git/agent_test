#!/bin/bash

set -e

export PATH="$HOME/.local/bin:$PATH"

# FastAPI 및 Streamlit 서버 시작 스크립트

# 기존 프로세스 종료 (PID 파일 사용)
echo "1. 기존 프로세스 종료 중..."
if [ -f app.pid ]; then
  PID=$(cat app.pid)
  if ps -p $PID > /dev/null; then
    echo "- Backend(PID: $PID) 종료 중..."
    kill $PID
  fi
  rm -f app.pid
fi

if [ -f ui.pid ]; then
  PID=$(cat ui.pid)
  if ps -p $PID > /dev/null; then
    echo "- Frontend(PID: $PID) 종료 중..."
    kill $PID
  fi
  rm -f ui.pid
fi

echo "2. 의존성 설치 중..."
uv sync

# 새 서버 시작
echo "3. 백엔드 서버(FastAPI) 시작 중..."
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8001 > app.log 2>&1 &
echo $! > app.pid

echo "4. 서비스 준비 상태 확인 중..."
while true; do
    # 1) 백엔드 헬스 체크
    HEALTH_JSON=$(curl -s http://localhost:8001/agent/health || echo '{"status":"waiting"}')
    HEALTH_STATUS=$(echo $HEALTH_JSON | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    
    if [ "$HEALTH_STATUS" != "healthy" ]; then
        echo -ne "\r[*] 백엔드 서비스 응답 대기 중 (Health Check)..."
        sleep 5
        continue
    fi

    # 2) 데이터 시딩 상태 확인
    STATUS_JSON=$(curl -s http://localhost:8001/agent/seed-status || echo '{"status":"waiting"}')
    STATUS=$(echo $STATUS_JSON | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    CURRENT=$(echo $STATUS_JSON | grep -o '"current":[0-9]*' | cut -d':' -f2)
    TOTAL=$(echo $STATUS_JSON | grep -o '"total":[0-9]*' | cut -d':' -f2)
    MESSAGE=$(echo $STATUS_JSON | grep -o '"message":"[^"]*"' | cut -d'"' -f4)

    if [ "$STATUS" = "completed" ]; then
        # 3) DB 데이터 존재 확인 (stats)
        STATS_JSON=$(curl -s http://localhost:8001/agent/stats || echo '{"count":0}')
        COUNT=$(echo $STATS_JSON | grep -o '"count":[0-9]*' | cut -d':' -f2)
        
        if [ "$COUNT" -gt 0 ] && [ -n "$COUNT" ]; then
            echo -e "\n[✓] 서비스 및 데이터 준비 완료! (총 $COUNT 개의 문서)"
            break
        else
            echo -ne "\r[*] 시딩은 완료되었으나 DB 데이터 확인 중 (Count: $COUNT)..."
        fi
    elif [ "$STATUS" = "error" ]; then
        echo -e "\n[!] 시딩 중 오류 발생: $MESSAGE"
        break
    elif [ "$STATUS" = "in_progress" ]; then
        if [ "$TOTAL" -gt 0 ] && [ -n "$TOTAL" ]; then
            PERCENT=$((CURRENT * 100 / TOTAL))
            echo -ne "\r[*] 시딩 진행 중: $CURRENT / $TOTAL ($PERCENT%) - $MESSAGE"
        else
            echo -ne "\r[*] 시딩 준비 중: $MESSAGE"
        fi
    else
        echo -ne "\r[*] 데이터 시딩 상태 확인 중..."
    fi
    sleep 5
done

echo -e "\n5. 프론트엔드 서버(Streamlit) 시작 중..."
export BACKEND_URL="http://localhost:8001"
nohup uv run streamlit run infra/frontend/ui.py --server.port 8002 > ui.log 2>&1 &
echo $! > ui.pid

echo -e "\n-------------------------------------------------------"
echo "서비스가 성공적으로 시작되었습니다."
echo "백엔드 접속: http://localhost:8001"
echo "프론트엔드 접속: http://localhost:8002"
echo "백엔드 로그: tail -f app.log"
echo "프론트엔드 로그: tail -f ui.log"
echo "-------------------------------------------------------"

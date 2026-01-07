from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from app.api.route.agent_routers import router as agent_router
from app.core.seed import seed_data_if_empty
from app.exceptions import AgentException, KnowledgeBaseException, ValidationException


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    # 시딩 작업이 오래 걸릴 수 있으므로 백그라운드 태스크로 실행하거나
    # 여기서는 진행 상황을 알 수 있도록 함. 
    # 동기 함수인 경우 루프에서 별도 스레드로 실행 권장
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, seed_data_if_empty)
    yield
    # 앱 종료 시 실행 (필요한 경우)


app = FastAPI(lifespan=lifespan)


@app.exception_handler(AgentException)
async def agent_exception_handler(request: Request, exc: AgentException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "AgentException",
            "message": exc.message,
            "details": exc.details
        },
    )


@app.exception_handler(KnowledgeBaseException)
async def knowledge_base_exception_handler(request: Request, exc: KnowledgeBaseException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "KnowledgeBaseException",
            "message": exc.message,
            "details": exc.details
        },
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "ValidationException",
            "message": exc.message,
            "details": exc.details
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "ValueError",
            "message": str(exc),
            "details": {}
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "details": {}
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"type": exc.__class__.__name__, "info": str(exc)}
        },
    )


app.include_router(agent_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

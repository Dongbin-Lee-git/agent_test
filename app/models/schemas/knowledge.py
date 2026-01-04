from typing import List, Dict, Any, Optional
from pydantic import Field
from .agent import BaseSchema


class AddKnowledgeRequest(BaseSchema):
    """지식 추가 요청 스키마"""
    documents: List[str] = Field(..., description="추가할 문서 리스트")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="문서별 메타데이터 리스트")


class KnowledgeResponse(BaseSchema):
    """지식 작업 응답 스키마"""
    status: str = Field(..., description="상태 (success/error)")
    message: str = Field(..., description="결과 메시지")


class StatsResponse(BaseSchema):
    """지식 베이스 통계 응답 스키마"""
    name: str = Field(..., description="콜렉션 이름")
    count: int = Field(..., description="문서 개수")
    metadata: Optional[Dict[str, Any]] = Field(None, description="콜렉션 메타데이터")

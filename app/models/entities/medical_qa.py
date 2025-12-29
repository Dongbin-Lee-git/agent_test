from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class MedicalQA(BaseModel):
    """
    의료 지식 베이스의 개별 문서를 나타내는 엔티티
    """
    id: Optional[str] = Field(None, description="문서 식별자")
    question: str = Field(..., description="질문 내용")
    answer: str = Field(..., description="답변 내용")
    source: Optional[str] = Field(None, description="출처 (예: Naver, Internal, Google)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

    class Config:
        from_attributes = True

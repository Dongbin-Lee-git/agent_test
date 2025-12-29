from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from app.models.schemas.agent import Message

class UserSession(BaseModel):
    """
    사용자 대화 세션을 나타내는 엔티티
    """
    session_id: str = Field(..., description="세션 고유 식별자")
    user_id: Optional[str] = Field(None, description="사용자 식별자 (로그인 시)")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    history: List[Message] = Field(default_factory=list, description="대화 내역")

    class Config:
        from_attributes = True

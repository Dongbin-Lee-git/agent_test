import os
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from dotenv import load_dotenv
from app.repository.client.base import BaseLLMClient

# 배포 환경(Kubernetes)에서는 ConfigMap/Secret으로 환경변수가 자동 주입되므로 .env 로드를 건너뜁니다.
# 로컬 개발 환경에서만 .env 파일을 읽어오도록 처리합니다.
if os.getenv("KUBERNETES_SERVICE_HOST") is None:
    load_dotenv()


class UpstageClient(BaseLLMClient):
    def __init__(self):
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.chat_model_name = os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro2")
        self.embedding_model_name = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
        self._chat_instance = None
        self._embedding_instance = None

    def get_chat_model(self) -> ChatUpstage:
        if self._chat_instance is None:
            self._chat_instance = ChatUpstage(api_key=self.api_key, model=self.chat_model_name)
        return self._chat_instance

    def get_embedding_model(self) -> UpstageEmbeddings:
        if self._embedding_instance is None:
            self._embedding_instance = UpstageEmbeddings(api_key=self.api_key, model=self.embedding_model_name)
        return self._embedding_instance

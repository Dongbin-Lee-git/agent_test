import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from app.repository.client.base import BaseSearchClient
from dotenv import load_dotenv

# 배포 환경(Kubernetes)에서는 ConfigMap/Secret으로 환경변수가 자동 주입되므로 .env 로드를 건너뜁니다.
# 로컬 개발 환경에서만 .env 파일을 읽어오도록 처리합니다.
if os.getenv("KUBERNETES_SERVICE_HOST") is None:
    load_dotenv()


class SerperSearchClient(BaseSearchClient):
    def __init__(self):
        self._search = GoogleSerperAPIWrapper()

    def search(self, query: str) -> str:
        return self._search.run(query)

from langchain_community.utilities import GoogleSerperAPIWrapper
from app.repository.client.base import BaseSearchClient

class SerperSearchClient(BaseSearchClient):
    def __init__(self):
        self._search = GoogleSerperAPIWrapper()

    def search(self, query: str) -> str:
        return self._search.run(query)

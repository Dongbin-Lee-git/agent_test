from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    @abstractmethod
    def get_chat_model(self) -> Any:
        pass

    @abstractmethod
    def get_embedding_model(self) -> Any:
        pass


class BaseSearchClient(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        pass

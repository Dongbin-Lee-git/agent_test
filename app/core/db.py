import logging
import os
import chromadb
from typing import Optional
from dotenv import load_dotenv

# Suppress all chromadb related logging before it starts
logging.getLogger("chromadb").setLevel(logging.ERROR)

load_dotenv()

logger = logging.getLogger("chroma")


class ChromaDBConfig:
    def __init__(self):
        self.mode = os.getenv("CHROMA_MODE", "local")  # "local" or "server"
        self.host = os.getenv("CHROMA_HOST", "localhost")
        self.port = int(os.getenv("CHROMA_PORT", "8000"))
        self.persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "upstage_embeddings")
        logger.info(f"ChromaDB Mode: {self.mode}, Path: {self.persist_path}, Host: {self.host}:{self.port}")


class ChromaDBConnection:
    _instance: Optional["ChromaDBConnection"] = None
    _client: Optional[chromadb.ClientAPI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            config = ChromaDBConfig()
            from chromadb.config import Settings

            if config.mode == "server":
                logger.info(f"Connecting to ChromaDB server at {config.host}:{config.port}")
                self._client = chromadb.HttpClient(
                    host=config.host,
                    port=config.port
                )
            else:
                logger.info(f"Using local PersistentClient at {config.persist_path}")
                self._client = chromadb.PersistentClient(
                    path=config.persist_path,
                    settings=Settings(anonymized_telemetry=False)
                )

    @property
    def client(self) -> chromadb.ClientAPI:
        return self._client

    def get_collection(self, collection_name: str = None):
        config = ChromaDBConfig()
        name = collection_name or config.collection_name
        return self._client.get_or_create_collection(
            name=name, metadata={"description": "Upstage Solar2 embeddings collection"}
        )


def get_chroma_client() -> chromadb.ClientAPI:
    """ChromaDB 클라이언트를 반환하는 의존성 함수"""
    connection = ChromaDBConnection()
    return connection.client


def get_chroma_collection(collection_name: str = None):
    """ChromaDB 컬렉션을 반환하는 의존성 함수"""
    connection = ChromaDBConnection()
    return connection.get_collection(collection_name)

from typing import List, Dict, Any
from app.exceptions import KnowledgeBaseException
from .embedding_service import EmbeddingService
from ..repository.vector.vector_repo import VectorRepository


class VectorService:
    def __init__(
        self, vector_repository: VectorRepository, embedding_service: EmbeddingService
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ):
        try:
            embeddings = self.embedding_service.create_embeddings(documents)
            self.vector_repository.add_documents(
                documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
            )
        except Exception as e:
            raise KnowledgeBaseException(f"Failed to add documents: {str(e)}")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            query_embedding = self.embedding_service.create_embedding(query)

            results = self.vector_repository.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0],
            }
        except Exception as e:
            raise KnowledgeBaseException(f"Search failed: {str(e)}")

    def delete_document(self, doc_id: str):
        self.vector_repository.delete_documents([doc_id])

    def get_collection_info(self) -> Dict[str, Any]:
        return self.vector_repository.get_collection_info()

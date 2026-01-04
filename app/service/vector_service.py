from typing import List, Dict, Any
from app.exceptions import KnowledgeBaseException
from app.models import MedicalQA
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

    def search(self, query: str, n_results: int = 5) -> List[MedicalQA]:
        try:
            query_embedding = self.embedding_service.create_embedding(query)

            results = self.vector_repository.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            qa_list = []
            for doc, meta, dist, doc_id in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    results["ids"][0] if "ids" in results else [None] * len(results["documents"][0])
            ):
                qa_list.append(MedicalQA(
                    id=doc_id,
                    document=doc,
                    metadata=meta
                ))

            return qa_list
        except Exception as e:
            raise KnowledgeBaseException(f"Search failed: {str(e)}")

    def delete_document(self, doc_id: str):
        self.vector_repository.delete_documents([doc_id])

    def get_collection_info(self) -> Dict[str, Any]:
        return self.vector_repository.get_collection_info()

from fastapi.params import Depends

from app.repository.vector.vector_repo import VectorRepository, ChromaDBRepository
from app.service.vector_service import VectorService
from app.service.embedding_service import EmbeddingService
from app.service.agent_service import AgentService
from app.service.agents.info_extractor_service import InfoExtractorService
from app.service.agents.knowledge_augmentor_service import KnowledgeAugmentorService
from app.service.agents.answer_gen_service import AnswerGenService


def get_vector_repository() -> VectorRepository:
    return ChromaDBRepository()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_vector_service(
        vector_repo: VectorRepository = Depends(get_vector_repository),
        embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> VectorService:
    return VectorService(
        vector_repository=vector_repo, embedding_service=embedding_service
    )


def get_info_extractor_service() -> InfoExtractorService:
    return InfoExtractorService()


def get_knowledge_augmentor_service() -> KnowledgeAugmentorService:
    return KnowledgeAugmentorService()


def get_answer_gen_service() -> AnswerGenService:
    return AnswerGenService()


def get_agent_service(
        vector_service: VectorService = Depends(get_vector_service),
        info_extractor_service: InfoExtractorService = Depends(get_info_extractor_service),
        knowledge_augmentor_service: KnowledgeAugmentorService = Depends(get_knowledge_augmentor_service),
        answer_gen_service: AnswerGenService = Depends(get_answer_gen_service),
) -> AgentService:
    return AgentService(
        vector_service=vector_service,
        info_extractor_service=info_extractor_service,
        knowledge_augmentor_service=knowledge_augmentor_service,
        answer_gen_service=answer_gen_service,
    )

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from app.service.agent_service import AgentService
from app.service.vector_service import VectorService
from app.models.exceptions import AgentNotFoundException, KnowledgeBaseException


class TestAgentService:
    @pytest.fixture
    def mock_vector_service(self):
        return Mock(spec=VectorService)

    @pytest.fixture
    def mock_upstage_client(self):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response from Solar LLM"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def agent_service(self, mock_vector_service, mock_upstage_client):
        with patch.dict(os.environ, {"UPSTAGE_API_KEY": "test-key"}):
            with patch(
                "app.service.agent_service.OpenAI", return_value=mock_upstage_client
            ):
                agent = AgentService(vector_service=mock_vector_service)
                return agent

    @pytest.mark.unit
    def test_agent_service_initialization_without_api_key(self, mock_vector_service):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="UPSTAGE_API_KEY environment variable is required"
            ):
                AgentService(vector_service=mock_vector_service)

    @pytest.mark.unit
    def test_add_knowledge_success(self, agent_service, mock_vector_service):
        test_documents = ["Test document 1", "Test document 2"]
        mock_vector_service.add_documents.return_value = None

        result = agent_service.add_knowledge(test_documents)

        assert result["status"] == "success"
        assert "Added 2 documents" in result["message"]
        mock_vector_service.add_documents.assert_called_once_with(test_documents, None)

    @pytest.mark.unit
    def test_add_knowledge_with_metadata(self, agent_service, mock_vector_service):
        test_documents = ["Test document"]
        test_metadata = [{"source": "test"}]

        result = agent_service.add_knowledge(test_documents, test_metadata)

        assert result["status"] == "success"
        mock_vector_service.add_documents.assert_called_once_with(
            test_documents, test_metadata
        )

    @pytest.mark.unit
    def test_add_knowledge_failure(self, agent_service, mock_vector_service):
        test_documents = ["Test document"]
        # VectorService.add_documents now raises KnowledgeBaseException on failure
        mock_vector_service.add_documents.side_effect = Exception("Vector service error")

        result = agent_service.add_knowledge(test_documents)

        assert result["status"] == "error"
        assert "Vector service error" in result["message"]

    @pytest.mark.unit
    def test_get_knowledge_stats(self, agent_service, mock_vector_service):
        expected_stats = {"name": "test_collection", "count": 5, "metadata": {}}
        mock_vector_service.get_collection_info.return_value = expected_stats

        result = agent_service.get_knowledge_stats()

        assert result == expected_stats
        mock_vector_service.get_collection_info.assert_called_once()

    @pytest.mark.unit
    def test_run_agent_not_found(self, agent_service):
        with pytest.raises(AgentNotFoundException):
            agent_service.run_agent("non_existent_agent", {"user_query": "hello"})


@pytest.mark.integration
class TestAgentServiceIntegration:
    """
    Integration tests that require actual services to be running.
    Run with: pytest -m integration
    """

    @pytest.fixture
    def agent_service_integration(self):
        """
        Creates a real AgentService for integration testing.
        Requires UPSTAGE_API_KEY and ChromaDB to be running.
        """
        if not os.getenv("UPSTAGE_API_KEY"):
            pytest.skip("UPSTAGE_API_KEY not found - skipping integration test")

        try:
            from app.deps import get_vector_repository, get_embedding_service
            vector_repo = get_vector_repository()
            embedding_service = get_embedding_service()
            vector_service = VectorService(vector_repo, embedding_service)
            return AgentService(vector_service=vector_service)
        except Exception as e:
            pytest.skip(f"Could not initialize AgentService: {e}")

    def test_full_agent_workflow(self, agent_service_integration):
        """Test the complete agent workflow with real services."""
        agent = agent_service_integration
        session_id = "test-session-agent-service"

        # Add test knowledge
        test_docs = [
            "Python is a programming language.",
            "FastAPI is a web framework for Python.",
        ]

        result = agent.add_knowledge(test_docs)
        assert result["status"] == "success"

        # Check knowledge stats
        stats = agent.get_knowledge_stats()
        assert stats["count"] > 0

        # Run agent - "super" agent requires session_id because of MemorySaver
        inputs = {"user_query": "Python이 뭐야?"}
        result = agent.run_agent("super", inputs, session_id=session_id)
        
        # Check MainState structure
        assert "answer_logs" in result
        assert len(result["answer_logs"]) > 0
        assert "user_query" in result
        assert result["user_query"] == "Python이 뭐야?"

    def test_agent_with_real_chromadb_connection(self, agent_service_integration):
        """Test that agent can connect to and use ChromaDB."""
        agent = agent_service_integration

        # Test getting collection info (this tests ChromaDB connection)
        stats = agent.get_knowledge_stats()
        assert "name" in stats
        assert "count" in stats
        assert isinstance(stats["count"], int)


# Test runner for backward compatibility with existing script
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

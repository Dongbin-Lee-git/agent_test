from typing import Optional, Dict

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.llm import get_solar_chat, get_upstage_embeddings
from app.service.vector_service import VectorService
from app.repository.client.search_client import SerperSearchClient


embedding_fn = get_upstage_embeddings()
solar_chat = get_solar_chat()
search_client = SerperSearchClient()

@tool
def add_to_medical_qa(content: str, config: RunnableConfig, metadata: Optional[Dict] = None) -> str:
    """
    Add new medical information to the knowledge base (ChromaDB).
    Use this to save useful information found from external sources.
    """
    print(f"\n[Tool: Add Knowledge] Adding content to DB...")
    print(f"  - Content snippet: {content[:100]}...")
    try:
        vector_service: VectorService = config["configurable"].get("vector_service")
        if not vector_service:
            return "Error: VectorService not found in config"
            
        vector_service.add_documents([content], [metadata or {"source": "google_search"}])
        print(f"[Tool: Add Knowledge] Successfully added.")
        return "Successfully added information to knowledge base."
    except Exception as e:
        print(f"[Tool: Add Knowledge] Error: {e}")
        return f"Error adding to knowledge base: {e}"

@tool
def google_search(query: str) -> str:
    """
    Search Google via Serper.dev for up-to-date medical information or news.
    Use this only when internal knowledge is insufficient.
    """
    print(f"\n[Tool: Google Search] Query: {query}")
    try:
        result = search_client.search(query)
        print(f"[Tool: Google Search] Result: {result[:200]}...")
        return result
    except Exception as e:
        print(f"[Tool: Google Search] Error: {e}")
        return f"Google Search Error: {e}"

@tool
def search_medical_qa(query: str, config: RunnableConfig) -> str:
    """
    Search medical QA database for relevant information.
    Returns a list of relevant QA pairs.
    """
    print(f"\n[Tool: Internal DB Search] Query: {query}")
    try:
        vector_service: VectorService = config["configurable"].get("vector_service")
        if not vector_service:
            return "Error: VectorService not found in config"

        results = vector_service.search(query, n_results=5)
        documents = results.get("documents", [])
        
        print(f"[Tool: Internal DB Search] Found {len(documents)} documents.")
        
        context_parts = []
        for i, doc in enumerate(documents):
            print(f"  - Document {i+1}: {doc[:100]}...")
            context_parts.append(f"Source {i+1}:\n{doc}")
        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[Tool: Internal DB Search] Error: {e}")
        return f"Search Error: {e}"


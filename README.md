# Upstage Gangwon Backend

FastAPI backend with agenic functionality using Upstage Solar2 embeddings and ChromaDB vector database.

## Features

- **User Management**: Basic user CRUD operations
- **Agent Service**: RAG-based question answering using Upstage Solar2
- **Vector Database**: ChromaDB for storing and retrieving embeddings
- **Embedding Service**: Upstage Solar2 embedding model integration

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   
   Required variables:
   - `UPSTAGE_API_KEY`: Your Upstage API key

3. **Start ChromaDB**:
   ```bash
   docker-compose -f infra/chroma/docker-compose.yml up -d
   ```

4. **Start the Server**:
   ```bash
   uvicorn main:app --reload
   ```

5. **Start Streamlit UI**:
   ```bash
   streamlit run frontend/ui.py
   ```

## API Endpoints

### Agent Endpoints

- `POST /agent/query`: Process a query using RAG
- `POST /agent/knowledge`: Add documents to knowledge base
- `GET /agent/stats`: Get knowledge base statistics
- `DELETE /agent/knowledge/{doc_id}`: Delete a document
- `GET /agent/health`: Health check

### User Endpoints

- `POST /users/`: Create a new user

## Testing

Run the test script to verify the agent functionality:

```bash
python test_agent.py
```

## Usage Example

```python
import requests

# Add knowledge to the agent
documents = [
    "FastAPI is a modern web framework for building APIs with Python.",
    "ChromaDB is an open-source vector database for AI applications."
]

response = requests.post("http://localhost:8000/agent/knowledge", 
                        json={"documents": documents})

# Query the agent
query_response = requests.post("http://localhost:8000/agent/query",
                              json={"query": "What is FastAPI?"})

print(query_response.json())
```

## Architecture

The system implements a Retrieval-Augmented Generation (RAG) pipeline:

1. **Embedding Service** (`app/service/embedding_service.py`): Uses Upstage Solar2 to create embeddings
2. **Vector Service** (`app/service/vector_service.py`): Manages ChromaDB operations
3. **Agent Service** (`app/service/agent_service.py`): Orchestrates the RAG pipeline
4. **API Routes** (`app/api/route/agent_routers.py`): FastAPI endpoints

## Infrastructure

- **ChromaDB**: Vector database running in Docker
- **Upstage Solar2**: Embedding and language model
- **FastAPI**: Web framework
- **Docker**: Container orchestration

## Tech Stack

LangGraph, Solar Pro2, Solar Embedding, Chroma DB, 
토큰량을 계산하기 위한 solar-pro-tokenzier (링크), 검색을 위한 Serper.dev 정도를 생각 중입니다.


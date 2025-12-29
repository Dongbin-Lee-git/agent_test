import os
import json
import logging
from typing import List, Dict, Any
from app.service.vector_service import VectorService
from app.repository.vector.vector_repo import ChromaDBRepository
from app.service.embedding_service import EmbeddingService

# 전역 로깅 설정 (콘솔 출력 보장)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("seed")

def load_medical_data(base_path: str = "resources/의료데이터") -> List[Dict[str, Any]]:
    documents = []
    if not os.path.exists(base_path):
        logger.warning(f"Path not found: {base_path}")
        return documents

    total_files = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    logger.debug(f"Loading medical data from: {file_path}")
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                        # JSON 형식에 따라 content 구성
                        # 예시에서 본 구조: question, answer
                        question = data.get("question", "")
                        answer = data.get("answer", "")
                        content = f"질문: {question}\n답변: {answer}"
                        
                        metadata = {
                            "source": file_path,
                            "qa_id": data.get("qa_id"),
                            "domain": data.get("domain"),
                            "q_type": data.get("q_type")
                        }
                        
                        documents.append({
                            "content": content,
                            "metadata": metadata,
                            "id": f"medical_{data.get('qa_id', file)}"
                        })
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {total_files} JSON files.")
    print(f"[*] Loaded {len(documents)} documents from {total_files} JSON files.")
    return documents

def seed_data_if_empty():
    repo = ChromaDBRepository()
    info = repo.get_collection_info()
    
    if info["count"] > 0:
        logger.info(f"Collection {info['name']} already has {info['count']} documents. Skipping seed.")
        print(f"[!] Collection {info['name']} already has {info['count']} documents. Skipping seed.")
        return

    logger.info(f"Collection {info['name']} is empty. Starting data seed...")
    print(f"[*] Collection {info['name']} is empty. Starting data seed...")
    
    medical_docs = load_medical_data()
    if not medical_docs:
        logger.warning("No medical data found to seed.")
        return

    api_key = os.getenv("UPSTAGE_API_KEY")
    if api_key:
        logger.info("UPSTAGE_API_KEY found. Proceeding with embedding and seeding.")
    else:
        logger.warning("UPSTAGE_API_KEY not found. Seeding might fail if embeddings are required.")

    embedding_service = EmbeddingService()
    vector_service = VectorService(repo, embedding_service)
    
    # 데이터를 배치로 나누어 삽입 (ChromaDB나 API 제한 고려)
    batch_size = 50 # 100에서 50으로 축소하여 안정성 확보 및 로그 빈도 증가
    total_docs = len(medical_docs)
    logger.info(f"Starting to add {total_docs} documents to ChromaDB in batches of {batch_size}...")
    print(f"[*] Starting to add {total_docs} documents to ChromaDB... This may take a while.")
    
    for i in range(0, total_docs, batch_size):
        batch = medical_docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        ids = [doc["id"] for doc in batch]
        
        try:
            vector_service.add_documents(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            current_count = i + len(batch)
            progress_msg = f"Progress: {current_count}/{total_docs} documents seeded ({(current_count/total_docs)*100:.1f}%)."
            logger.info(progress_msg)
            print(f"[*] {progress_msg}")
        except Exception as e:
            logger.error(f"Error seeding batch at index {i}: {e}")
            print(f"[!] Error seeding batch at index {i}: {e}")
            raise e

    logger.info(f"Data seeding to {info['name']} completed successfully. Total: {total_docs} documents.")
    print(f"[✓] Data seeding completed successfully. Total: {total_docs} documents.")

import os
import json
import unicodedata
import logging
import zipfile
import gdown
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

SEED_STATUS_FILE = "/tmp/seed_status.json"

def update_seed_status(status: str, current: int = 0, total: int = 0, message: str = ""):
    try:
        with open(SEED_STATUS_FILE, "w") as f:
            json.dump({
                "status": status,
                "current": current,
                "total": total,
                "message": message
            }, f)
    except Exception as e:
        logger.error(f"Failed to update seed status: {e}")

def get_seed_status():
    if not os.path.exists(SEED_STATUS_FILE):
        return {"status": "not_started", "current": 0, "total": 0, "message": "No seed status found."}
    try:
        with open(SEED_STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "current": 0, "total": 0, "message": str(e)}

def download_and_extract_data():
    """구글 드라이브에서 데이터를 다운로드하고 압축을 해제합니다."""
    url = os.getenv("SEED_URL")
    output = "resources/의료데이터.zip"
    extract_to = "resources"

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    try:
        if not os.path.exists(output):
            logger.info("Downloading data from SEED_URL...")
            gdown.download(url, output, quiet=False)
        
        logger.info(f"Extracting {output} to {extract_to}...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # zipfile은 한글 파일명을 인식하지 못하고 cp437로 처리하는 경우가 많음
                try:
                    # Mac/Linux UTF-8 시도
                    filename = file_info.filename.encode('cp437').decode('utf-8')
                except:
                    try:
                        # Windows CP949 시도
                        filename = file_info.filename.encode('cp437').decode('cp949')
                    except:
                        filename = file_info.filename
                
                # NFD(Mac) -> NFC 변환
                filename = unicodedata.normalize('NFC', filename)
                
                target_path = os.path.join(extract_to, filename)
                if file_info.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        logger.info("Data extraction completed.")
    except Exception as e:
        logger.error(f"Error downloading or extracting data: {e}")

def load_medical_data(base_path: str = "resources/의료데이터") -> List[Dict[str, Any]]:
    # 데이터가 없으면 다운로드 시도
    if not os.path.exists(base_path) or not os.listdir(base_path):
        logger.info(f"Data directory {base_path} is missing or empty. Attempting to download...")
        download_and_extract_data()

    documents = []
    if not os.path.exists(base_path):
        logger.warning(f"Path still not found after download attempt: {base_path}")
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
    return documents

def seed_data_if_empty():
    embedding_service = EmbeddingService()
    repo = ChromaDBRepository()
    vector_service = VectorService(repo, embedding_service)
    
    info = vector_service.get_collection_info()
    
    # 1개만 있는 경우는 이전 테스트의 잔재일 수 있으므로 10,000개 미만이면 시딩 시도
    if info["count"] >= 10000:
        logger.info(f"Collection {info['name']} already has {info['count']} documents. Skipping seed.")
        update_seed_status("completed", info["count"], info["count"], "Already seeded.")
        return

    logger.info(f"Collection {info['name']} has {info['count']} documents. Starting data seed to reach target...")
    update_seed_status("in_progress", 0, 0, "Loading data...")
    
    medical_docs = load_medical_data()
    if not medical_docs:
        logger.warning("No medical data found to seed.")
        update_seed_status("completed", info["count"], info["count"], "No data found to seed.")
        return

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.warning("UPSTAGE_API_KEY not found. Seeding might fail.")

    # 데이터를 배치로 나누어 삽입
    batch_size = 1000
    total_docs = len(medical_docs)
    logger.info(f"Starting to add {total_docs} documents to ChromaDB in batches of {batch_size}...")
    update_seed_status("in_progress", 0, total_docs, "Starting batch insertion...")
    
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
            update_seed_status("in_progress", current_count, total_docs, progress_msg)
        except Exception as e:
            logger.error(f"Error seeding batch at index {i}: {e}")
            update_seed_status("error", i, total_docs, str(e))
            raise e

    logger.info(f"Data seeding to {info['name']} completed successfully. Total: {total_docs} documents.")
    update_seed_status("completed", total_docs, total_docs, "Seeding completed successfully.")

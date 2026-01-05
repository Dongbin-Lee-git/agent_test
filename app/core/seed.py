import os
import json
import logging
import zipfile
import unicodedata
import gdown
from pathlib import Path
from typing import Iterator, Dict, Any, List

from app.models.entities.medical_qa import MedicalQA
from app.service.vector_service import VectorService
from app.repository.vector.vector_repo import ChromaDBRepository
from app.service.embedding_service import EmbeddingService
from app.core.db import ChromaDBConfig

# 로거 설정
logger = logging.getLogger("seed")


class SeedManager:
    """의료 데이터 다운로드 및 벡터 DB 시딩을 담당하는 매니저 클래스"""

    def __init__(self):
        self.seed_status_file = Path("logs/seed_status.json")
        self.resource_dir = Path("resources")
        self.data_dir = self.resource_dir / "의료데이터"
        self.zip_path = self.resource_dir / "의료데이터.zip"
        self.seed_url = os.getenv("SEED_URL")

        # 서비스 초기화
        self.vector_service = VectorService(ChromaDBRepository(), EmbeddingService())

    def get_status(self) -> Dict[str, Any]:
        """현재 시딩 상태를 반환합니다."""
        if not self.seed_status_file.exists():
            return {"status": "not_started", "current": 0, "total": 0, "message": "Ready"}
        try:
            with open(self.seed_status_file, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _update_status(self, status: str, current: int = 0, total: int = 0, message: str = ""):
        """상태 파일 업데이트 (내부용)"""
        try:
            with open(self.seed_status_file, "w") as f:
                json.dump({
                    "status": status,
                    "current": current,
                    "total": total,
                    "message": message
                }, f)
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def _download_and_extract(self):
        """데이터 다운로드 및 압축 해제"""
        if not self.resource_dir.exists():
            self.resource_dir.mkdir(parents=True)

        # 다운로드
        if not self.zip_path.exists():
            if not self.seed_url:
                raise ValueError("SEED_URL environment variable is not set.")
            logger.info("Downloading data...")
            gdown.download(self.seed_url, str(self.zip_path), quiet=False)

        # 압축 해제
        logger.info(f"Extracting {self.zip_path}...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # 한글 파일명 인코딩 보정 (cp437 -> cp949/utf-8 -> NFC)
                try:
                    filename = file_info.filename.encode('cp437').decode('cp949')
                except:
                    filename = file_info.filename

                filename = unicodedata.normalize('NFC', filename)
                # target_path 보정: '의료데이터/' 가 포함된 경우 이를 self.resource_dir 기준으로 정규화
                if filename.startswith('의료데이터/'):
                    target_path = self.resource_dir / filename
                else:
                    # 압축 파일 내에 최상위 폴더가 없는 경우 '의료데이터' 폴더 안에 넣도록 유도하거나, 
                    # 현재 코드 구조상 self.data_dir가 'resources/의료데이터'이므로 
                    # filename 자체가 '의료데이터'로 시작하지 않는 경우를 처리
                    target_path = self.resource_dir / filename

                if file_info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        # 만약 압축 해제 후 self.data_dir 가 존재하지 않는다면, 
        # 자소 분리된 이름의 폴더가 생성되었을 가능성이 큼. 이를 '의료데이터'로 이름 변경 시도
        extracted_dirs = [d for d in self.resource_dir.iterdir() if d.is_dir() and d.name != "__MACOSX"]
        if not self.data_dir.exists() and extracted_dirs:
            # 가장 유력한 후보(첫 번째 폴더)를 '의료데이터'로 변경
            # (자소 분리 문제로 인해 self.data_dir.exists()가 False인 상황 가정)
            candidate = extracted_dirs[0]
            logger.info(f"Renaming {candidate.name} to {self.data_dir.name}")
            candidate.rename(self.data_dir)

    def _load_documents_generator(self) -> Iterator[MedicalQA]:
        """JSON 파일을 읽어 문서를 하나씩 반환하는 제너레이터 (메모리 절약)"""
        if not self.data_dir.exists():
            self._download_and_extract()

        logger.info(f"Loading documents from {self.data_dir}")
        count = 0
        for file_path in self.data_dir.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    # "Q: ..., A: ..." 형식으로 document 생성
                    document = f"Q: {data.get('question', '')}\nA: {data.get('answer', '')}"

                    # 나머지 정보들을 metadata 텍스트로도 포함
                    extra_info = f"Domain: {data.get('domain', 'N/A')}, Type: {data.get('q_type', 'N/A')}"

                    yield MedicalQA(
                        id=f"medical_{data.get('qa_id', file_path.stem)}",
                        document=document,
                        metadata={"extra_info": extra_info}
                    )
                    count += 1
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")

        logger.info(f"Total documents loaded from generator: {count}")

    def run(self):
        """시딩 프로세스 메인 실행 함수"""
        try:
            # DB 설정 정보 로그 출력
            from app.core.db import ChromaDBConfig
            config = ChromaDBConfig()
            logger.info(f"Starting seeding process. Mode: {config.mode}, Host: {config.host}, Port: {config.port}")
            
            # 1. 데이터 준비 및 파일 카운팅 (먼저 수행)
            self._update_status("in_progress", 0, 0, "Checking file integrity...")

            # 파일 시스템의 데이터를 먼저 로드하여 전체 개수 파악
            documents = list(self._load_documents_generator())
            total_docs = len(documents)

            if total_docs == 0:
                self._update_status("completed", 0, 0, "No data files found.")
                return

            # 2. DB 상태 확인 및 비교
            info = self.vector_service.get_collection_info()
            current_db_count = info["count"]

            # 파일 개수와 DB 개수가 같거나 DB가 더 많으면(중복 포함 등) 완료된 것으로 간주
            if current_db_count >= total_docs:
                logger.info(f"Already seeded correctly (DB: {current_db_count}, Files: {total_docs}). Skipping.")
                self._update_status("completed", current_db_count, total_docs, "Already seeded.")
                return

            logger.info(f"Seeding required. DB has {current_db_count}, but files have {total_docs}.")

            # 3. 배치 처리 및 삽입
            batch_size = 100
            self._update_status("in_progress", current_db_count, total_docs, "Inserting vectors...")

            # 효율성을 위해 전체를 다시 넣되, 이미 존재하는 ID는 ChromaDB가 처리(upsert/ignore 정책에 따름)
            # 여기서는 단순화를 위해 리스트 순회
            for i in range(0, total_docs, batch_size):
                batch: List[MedicalQA] = documents[i: i + batch_size]

                self.vector_service.add_documents(
                    documents=[d.document for d in batch],
                    metadatas=[d.metadata for d in batch],
                    ids=[d.id for d in batch]
                )

                current = min(i + len(batch), total_docs)
                self._update_status("in_progress", current, total_docs,
                                    f"Seeding... {int(current / total_docs * 100)}%")
                logger.info(f"Seeded batch {current}/{total_docs}")

            self._update_status("completed", total_docs, total_docs, "Seeding completed.")
            logger.info("Seeding finished successfully.")

        except Exception as e:
            logger.exception("Seeding failed")
            self._update_status("error", 0, 0, str(e))
            raise e


# 싱글톤 인스턴스 또는 함수 인터페이스 제공
seed_manager = SeedManager()


def get_seed_status():
    return seed_manager.get_status()


def seed_data_if_empty():
    seed_manager.run()

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logger import logger


class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, file: Union[str, object]) -> str:
        return file if isinstance(file, str) else file.name

    def validate_files(self, files: List) -> None:
        """Validate individual and total size of uploaded files."""
        total_size = 0
        for f in files:
            path = self._get_path(f)
            size = os.path.getsize(path)
            if size > constants.MAX_FILE_SIZE:
                raise ValueError(
                    f"File '{os.path.basename(path)}' exceeds the {constants.MAX_FILE_SIZE // 1024 // 1024}MB per-file limit"
                )
            total_size += size
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit")

    def process(self, files: List) -> List[Document]:
        """Process files with caching for subsequent queries."""
        self.validate_files(files)
        all_chunks: List[Document] = []
        seen_hashes: set = set()
        failed = 0

        for file in files:
            path = self._get_path(file)
            try:
                with open(path, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.json"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {path}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {path}")
                    chunks = self._process_file(path)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                failed += 1

        if failed == len(files):
            raise RuntimeError("All uploaded files failed to process. Check the logs for details.")

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, path: str) -> List[Document]:
        """Process a single file using Docling."""
        allowed = tuple(constants.ALLOWED_TYPES)
        if not path.endswith(allowed):
            logger.warning(f"Skipping unsupported file type: {path}")
            return []

        converter = DocumentConverter()
        markdown = converter.convert(path).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List[Document], cache_path: Path) -> None:
        data = {
            "timestamp": datetime.now().timestamp(),
            "chunks": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in data["chunks"]]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)

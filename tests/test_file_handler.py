import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def processor(tmp_path):
    with patch("document_processor.file_handler.settings") as mock_settings:
        mock_settings.CACHE_DIR = str(tmp_path)
        mock_settings.CACHE_EXPIRE_DAYS = 7
        from document_processor.file_handler import DocumentProcessor
        proc = DocumentProcessor()
    proc.cache_dir = tmp_path
    return proc


class TestGetPath:
    def test_string_path_returned_as_is(self, processor):
        assert processor._get_path("/some/file.pdf") == "/some/file.pdf"

    def test_file_object_returns_name(self, processor):
        mock_file = MagicMock()
        mock_file.name = "/some/file.pdf"
        assert processor._get_path(mock_file) == "/some/file.pdf"


class TestGenerateHash:
    def test_is_deterministic(self, processor):
        content = b"hello world"
        assert processor._generate_hash(content) == processor._generate_hash(content)

    def test_different_content_produces_different_hash(self, processor):
        assert processor._generate_hash(b"abc") != processor._generate_hash(b"xyz")

    def test_returns_64_char_hex_string(self, processor):
        result = processor._generate_hash(b"test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestCacheSaveLoad:
    def test_roundtrip_preserves_content_and_metadata(self, processor, tmp_path):
        docs = [Document(page_content="hello", metadata={"source": "test.pdf"})]
        cache_path = tmp_path / "cache.json"
        processor._save_to_cache(docs, cache_path)
        loaded = processor._load_from_cache(cache_path)
        assert len(loaded) == 1
        assert loaded[0].page_content == "hello"
        assert loaded[0].metadata == {"source": "test.pdf"}

    def test_cache_file_is_valid_json(self, processor, tmp_path):
        docs = [Document(page_content="test", metadata={})]
        cache_path = tmp_path / "cache.json"
        processor._save_to_cache(docs, cache_path)
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "chunks" in data
        assert "timestamp" in data

    def test_roundtrip_multiple_docs(self, processor, tmp_path):
        docs = [
            Document(page_content=f"chunk {i}", metadata={"page": i}) for i in range(5)
        ]
        cache_path = tmp_path / "multi.json"
        processor._save_to_cache(docs, cache_path)
        loaded = processor._load_from_cache(cache_path)
        assert len(loaded) == 5
        assert [d.page_content for d in loaded] == [f"chunk {i}" for i in range(5)]


class TestIsCacheValid:
    def test_returns_false_when_file_missing(self, processor, tmp_path):
        assert processor._is_cache_valid(tmp_path / "nonexistent.json") is False

    def test_returns_true_for_fresh_cache(self, processor, tmp_path):
        cache_path = tmp_path / "fresh.json"
        cache_path.write_text("{}")
        assert processor._is_cache_valid(cache_path) is True

    def test_returns_false_for_expired_cache(self, processor, tmp_path):
        from datetime import datetime, timedelta
        import time
        cache_path = tmp_path / "old.json"
        cache_path.write_text("{}")
        expired_time = (datetime.now() - timedelta(days=10)).timestamp()
        os.utime(cache_path, (expired_time, expired_time))
        assert processor._is_cache_valid(cache_path) is False


class TestValidateFiles:
    def test_raises_when_single_file_too_large(self, processor):
        from config import constants
        mock_file = MagicMock()
        mock_file.name = "/fake/large.pdf"
        with patch("os.path.getsize", return_value=constants.MAX_FILE_SIZE + 1):
            with pytest.raises(ValueError, match="per-file limit"):
                processor.validate_files([mock_file])

    def test_raises_when_total_size_exceeded(self, processor):
        from config import constants
        files = [MagicMock() for _ in range(3)]
        for i, f in enumerate(files):
            f.name = f"/fake/file{i}.pdf"
        per_file = constants.MAX_TOTAL_SIZE // 2 + 1
        with patch("os.path.getsize", return_value=per_file):
            with pytest.raises(ValueError, match="Total size"):
                processor.validate_files(files)

    def test_passes_for_valid_files(self, processor):
        mock_file = MagicMock()
        mock_file.name = "/fake/small.pdf"
        with patch("os.path.getsize", return_value=1024):
            processor.validate_files([mock_file])  # should not raise

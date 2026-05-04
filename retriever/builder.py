from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
from utils.logger import logger


class RetrieverBuilder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    def build_hybrid_retriever(self, docs) -> EnsembleRetriever:
        """Build a hybrid retriever using BM25 and vector-based retrieval."""
        try:
            # Ephemeral vector store — no persist_directory to avoid stale data across sessions
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
            )
            logger.info("Vector store created successfully")

            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully")

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            logger.info("Hybrid retriever created successfully")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise

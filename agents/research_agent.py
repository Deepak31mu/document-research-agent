from openai import OpenAI
from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings
from utils.logger import logger


class ResearchAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_id = "gpt-4o-mini"
        self.max_tokens = 1500
        self.temperature = 0.3

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        return f"""You are an AI assistant designed to provide precise and factual answers based on the given context.

**Instructions:**
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.

**Question:**
\"\"\"
{question}
\"\"\"

**Context:**
\"\"\"
{context}
\"\"\"

**Provide your answer below:**"""

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join(doc.page_content for doc in documents)
        logger.info(f"Research: {len(documents)} docs, context length {len(context)}")

        prompt = self.generate_prompt(question, context)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        try:
            llm_response = response.choices[0].message.content.strip()
        except (IndexError, AttributeError) as e:
            logger.error(f"Unexpected response structure: {e}")
            llm_response = ""

        draft_answer = (
            self.sanitize_response(llm_response)
            if llm_response
            else "I cannot answer this question based on the provided documents."
        )
        logger.info("Research agent produced an answer")
        return {"draft_answer": draft_answer, "context_used": context}

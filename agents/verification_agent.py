from openai import OpenAI
from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings
from utils.logger import logger

_DEFAULT_REPORT: Dict = {
    "Supported": "NO",
    "Unsupported Claims": [],
    "Contradictions": [],
    "Relevant": "NO",
    "Additional Details": "",
}


class VerificationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_id = "gpt-4o-mini"
        self.max_tokens = 500
        self.temperature = 0.0

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        return f"""You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

**Instructions:**
- Verify the following answer against the provided context.
- Check for:
  1. Direct/indirect factual support (YES/NO)
  2. Unsupported claims (list any if present)
  3. Contradictions (list any if present)
  4. Relevance to the question (YES/NO)
- Respond in the exact format specified below.

**Format:**
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information]

**Answer:**
\"\"\"
{answer}
\"\"\"

**Context:**
\"\"\"
{context}
\"\"\"

**Respond ONLY with the above format.**"""

    def parse_verification_response(self, response_text: str) -> Dict:
        try:
            verification = {}
            for line in response_text.split('\n'):
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip().title()
                value = value.strip()
                if key not in _DEFAULT_REPORT:
                    continue
                if key in {"Unsupported Claims", "Contradictions"}:
                    if value.startswith('[') and value.endswith(']'):
                        items = [i.strip().strip('"').strip("'") for i in value[1:-1].split(',') if i.strip()]
                        verification[key] = items
                    else:
                        verification[key] = []
                elif key == "Additional Details":
                    verification[key] = value
                else:
                    verification[key] = value.upper()
            for key, default in _DEFAULT_REPORT.items():
                verification.setdefault(key, default)
            return verification
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        unsupported = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        details = verification.get("Additional Details", "")
        lines = [
            f"**Supported:** {verification.get('Supported', 'NO')}",
            f"**Unsupported Claims:** {', '.join(unsupported) if unsupported else 'None'}",
            f"**Contradictions:** {', '.join(contradictions) if contradictions else 'None'}",
            f"**Relevant:** {verification.get('Relevant', 'NO')}",
            f"**Additional Details:** {details if details else 'None'}",
        ]
        return "\n".join(lines)

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        context = "\n\n".join(doc.page_content for doc in documents)
        logger.info(f"Verification: {len(documents)} docs, answer length {len(answer)}")

        prompt = self.generate_prompt(answer, context)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        try:
            llm_response = response.choices[0].message.content.strip()
        except (IndexError, AttributeError) as e:
            logger.error(f"Unexpected response structure: {e}")
            report = {**_DEFAULT_REPORT, "Additional Details": "Invalid response structure."}
            return {"verification_report": self.format_verification_report(report), "context_used": context}

        if not llm_response:
            logger.warning("LLM returned empty response")
            report = {**_DEFAULT_REPORT, "Additional Details": "Empty response from model."}
        else:
            report = self.parse_verification_response(llm_response)
            if report is None:
                logger.warning("Failed to parse verification response")
                report = {**_DEFAULT_REPORT, "Additional Details": "Failed to parse model response."}

        verification_report = self.format_verification_report(report)
        logger.info("Verification complete")
        return {"verification_report": verification_report, "context_used": context}

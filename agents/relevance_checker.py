from openai import OpenAI
from config.settings import settings
from utils.logger import logger


class RelevanceChecker:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_id = "gpt-4o-mini"

    def check(self, question: str, retriever, k: int = 3) -> str:
        """
        Retrieve top-k chunks and classify whether the question can be answered.
        Returns "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever")
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

**Important:** If the passages mention or reference the topic or timeframe in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

**Question:**
\"\"\"
{question}
\"\"\"

**Passages:**
\"\"\"
{document_content}
\"\"\"

**Respond ONLY with one of: CAN_ANSWER, PARTIAL, NO_MATCH**"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            return "NO_MATCH"

        try:
            llm_response = response.choices[0].message.content.strip().upper()
        except (IndexError, AttributeError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            logger.debug(f"Invalid label '{llm_response}', defaulting to NO_MATCH")
            return "NO_MATCH"

        logger.debug(f"Relevance classification: {llm_response}")
        return llm_response

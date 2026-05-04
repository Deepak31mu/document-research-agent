from openai import OpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class RelevanceChecker:
    def __init__(self):
        self.model_id = "gpt-4o-mini"

    def check(self, question: str, retriever, k=3) -> str:
        """
        Retrieve the top-k document chunks, then classify whether the question
        can be answered: returns "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever. Classifying as NO_MATCH.")
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.

        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "NO_MATCH"

        try:
            llm_response = response.choices[0].message.content.strip().upper()
            logger.debug(f"LLM response: {llm_response}")
        except (IndexError, AttributeError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return "NO_MATCH"

        print(f"Checker response: {llm_response}")

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            return "NO_MATCH"

        logger.debug(f"Classification recognized as '{llm_response}'.")
        return llm_response

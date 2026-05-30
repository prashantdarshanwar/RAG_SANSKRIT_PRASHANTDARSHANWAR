import logging

logger = logging.getLogger("SanskritRAG")


class PromptTemplate:
    """
    Builds the RAG prompt using TinyLlama-1.1B-Chat-v1.0's required
    Zephyr chat format: <|system|> ... </s> <|user|> ... </s> <|assistant|>

    Without this format TinyLlama ignores instructions and echoes the prompt.
    """

    SYSTEM_MESSAGE = (
        "You are an expert Sanskrit and Hindi language assistant. "
        "Answer the user's question using ONLY the passages provided. "
        "Give a concise, direct answer in the same language as the question. "
        "If the answer is not in the passages, say: "
        "\"The answer is not available in the provided document.\""
    )

    @staticmethod
    def build_prompt(query, context_chunks):
        """
        Assemble a TinyLlama-compatible chat prompt.

        Args:
            query (str): User's question (Sanskrit or English)
            context_chunks (list[str]): Retrieved text passages
        Returns:
            str: Formatted prompt with TinyLlama chat tokens
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            if not isinstance(context_chunks, list):
                raise TypeError(
                    f"context_chunks must be a list, got {type(context_chunks).__name__}"
                )

            valid_chunks = [
                c for c in context_chunks
                if isinstance(c, str) and c.strip()
            ]

            if not valid_chunks:
                logger.warning("All context chunks are empty. LLM will have no context.")

            numbered = "\n\n".join(
                f"[Passage {i+1}]\n{chunk.strip()}"
                for i, chunk in enumerate(valid_chunks)
            )

            # TinyLlama Zephyr chat format — required for instruction-following
            prompt = (
                f"<|system|>\n{PromptTemplate.SYSTEM_MESSAGE}</s>\n"
                f"<|user|>\n"
                f"Passages:\n{numbered}\n\n"
                f"Question: {query}</s>\n"
                f"<|assistant|>\n"
            )

            logger.info(
                f"Prompt built. Passages: {len(valid_chunks)} | "
                f"Prompt length: {len(prompt)} chars."
            )
            return prompt

        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            raise RuntimeError(f"build_prompt error: {e}") from e
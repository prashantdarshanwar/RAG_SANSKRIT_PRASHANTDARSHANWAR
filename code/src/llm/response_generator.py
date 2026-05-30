import logging
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.llm.model_loader import LLMModel
from src.llm.prompt_template import PromptTemplate

logger = logging.getLogger("SanskritRAG")


class ResponseGenerator:
    """
    Orchestrates end-to-end answer generation:
    Retrieve → Build Prompt → Generate Response

    Accepts a shared EmbeddingModel so the model is never loaded more than once
    across the entire pipeline.
    """

    def __init__(self, embedding_model: EmbeddingModel = None):
        try:
            logger.info("Initializing ResponseGenerator...")

            # ✅ One shared EmbeddingModel flows through:
            #    SanskritRAGPipeline → ResponseGenerator → Retriever
            #    The model is loaded exactly once for the entire app session.
            shared_model = embedding_model or EmbeddingModel()

            self.retriever = Retriever(embedding_model=shared_model)
            self.llm = LLMModel()

            logger.info("ResponseGenerator initialized successfully.")

        except Exception as e:
            logger.error(f"ResponseGenerator initialization failed: {e}")
            raise RuntimeError(f"ResponseGenerator init error: {e}") from e

    def generate_answer(self, query, top_k=3):
        """
        Full RAG answer generation for a user query.

        Args:
            query (str): User's question (Sanskrit or transliterated)
            top_k (int): Number of context chunks to retrieve
        Returns:
            dict: {query, retrieved_chunks, response}
        """
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            logger.info(f"Generating answer for: '{query}' | top_k={top_k}")

            # Step 1: Retrieve
            retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k)

            if not retrieved_chunks:
                logger.warning("No chunks retrieved — response will be generic.")

            # Step 2: Prompt
            prompt = PromptTemplate.build_prompt(query, retrieved_chunks)

            # Step 3: Generate
            response = self.llm.generate_response(prompt)

            logger.info("Answer generation complete.")
            return {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "response": response
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Answer generation failed for query '{query}': {e}")
            raise RuntimeError(f"generate_answer error: {e}") from e


# Testing
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    setup_logger()

    try:
        generator = ResponseGenerator()
        result = generator.generate_answer(query="अन्तिमे श्लोके लेखकः किम् उपदेशं ददाति ?", top_k=3)

        print("\n===== QUERY =====")
        print(result["query"])
        print("\n===== RETRIEVED CHUNKS =====")
        for i, chunk in enumerate(result["retrieved_chunks"]):
            print(f"\nChunk {i+1}:\n{chunk}")
        print("\n===== RESPONSE =====")
        print(result["response"])

    except Exception as e:
        print(f"Error: {e}")
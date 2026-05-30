import os
import json
import time
import logging

from src.data_ingestion.loader import load_document
from src.data_ingestion.text_cleaner import clean_text
from src.preprocessing.chunking import create_chunks
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore
from src.llm.response_generator import ResponseGenerator
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker

logger = logging.getLogger("SanskritRAG")


class SanskritRAGPipeline:
    """
    End-to-End Sanskrit RAG Pipeline.

    Single EmbeddingModel instance shared across build and query:
      SanskritRAGPipeline.embedding_model
        → passed into ResponseGenerator
          → passed into Retriever
    Result: model loaded exactly ONCE per app session.
    """

    ARTIFACT_DIR   = "artifacts"
    FAISS_PATH     = "artifacts/faiss_index"
    METRICS_PATH   = "artifacts/metrics"
    RESPONSES_PATH = "artifacts/responses"
    LOG_PATH       = "artifacts/logs"

    def __init__(self):
        try:
            setup_logger(log_dir=self.LOG_PATH)
            logger.info("Initializing SanskritRAGPipeline...")

            for path in [self.FAISS_PATH, self.METRICS_PATH,
                         self.RESPONSES_PATH, self.LOG_PATH]:
                os.makedirs(path, exist_ok=True)

            # ✅ One EmbeddingModel for the entire pipeline lifetime
            self.embedding_model    = EmbeddingModel()
            self.vector_store       = VectorStore()
            self.metrics            = MetricsTracker()

            # ResponseGenerator receives the shared embedding_model —
            # it passes it down to Retriever so it is never loaded again
            self.response_generator = ResponseGenerator(
                embedding_model=self.embedding_model
            )

            logger.info("SanskritRAGPipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise RuntimeError(f"SanskritRAGPipeline init error: {e}") from e

    def build_pipeline(self, document_path):
        """
        Ingest a document and build the FAISS index.
        Transliteration intentionally removed — original Devanagari/Sanskrit
        is stored in chunks so the LLM receives readable context.
        The multilingual MiniLM embedding model handles Devanagari natively.
        """
        logger.info("===== STARTING PIPELINE BUILD =====")

        try:
            self.metrics.start_timer()

            logger.info(f"Step 1 — Loading: {document_path}")
            raw_text = load_document(document_path)

            logger.info("Step 2 — Cleaning...")
            cleaned_text = clean_text(raw_text)

            # No transliteration — original script preserved for LLM context
            logger.info("Step 3 — Chunking...")
            chunks = create_chunks(cleaned_text, chunk_size=100, overlap=20)
            logger.info(f"Chunks created: {len(chunks)}")

            if not chunks:
                raise ValueError("No chunks created. Check document content.")

            logger.info("Step 4 — Embedding...")
            # ✅ Uses shared embedding_model — same instance as Retriever uses
            embeddings = self.embedding_model.generate_embeddings(chunks)

            logger.info("Step 5 — Indexing...")
            self.vector_store.create_index(embeddings, chunks)
            self.vector_store.save_index(save_path=self.FAISS_PATH)

            # ✅ Reload the retriever's vector store so it sees the new index
            # without needing to restart the app
            self.response_generator.retriever.vector_store.load_index(
                load_path=self.FAISS_PATH
            )

            self.metrics.stop_timer()
            latency = self.metrics.get_latency()
            self._save_build_artifact(document_path, len(chunks), latency)

            logger.info(f"===== BUILD COMPLETE ({latency}s) =====")

        except FileNotFoundError as e:
            logger.error(f"Document not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Build data error: {e}")
            raise
        except Exception as e:
            logger.error(f"Build failed: {e}")
            raise RuntimeError(f"build_pipeline error: {e}") from e

    def ask_question(self, query, top_k=3):
        """
        Query the RAG system. Uses the pre-loaded, shared response_generator.
        """
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            logger.info(f"Query: '{query}' | top_k={top_k}")
            self.metrics.start_timer()

            result = self.response_generator.generate_answer(
                query=query, top_k=top_k
            )

            self.metrics.stop_timer()
            latency = self.metrics.get_latency()

            accuracy = MetricsTracker.calculate_retrieval_accuracy(
                result["retrieved_chunks"],
                expected_keyword=query.split()[0]
            )
            self.metrics.record_run(
                query=query, latency=latency,
                accuracy=accuracy, num_chunks=len(result["retrieved_chunks"])
            )
            self.metrics.save_metrics_artifact(save_path=self.METRICS_PATH)
            self._save_response_artifact(result)

            logger.info(f"Done. Latency: {latency}s")
            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"ask_question failed: {e}")
            raise RuntimeError(f"ask_question error: {e}") from e

    def _save_build_artifact(self, document_path, num_chunks, latency):
        try:
            summary = {
                "document": document_path,
                "chunks_created": num_chunks,
                "build_latency_seconds": latency,
                "faiss_index_path": self.FAISS_PATH,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(os.path.join(self.METRICS_PATH, "build_summary.json"),
                      "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save build artifact: {e}")

    def _save_response_artifact(self, result):
        try:
            entry = {
                "query":    result["query"],
                "response": result["response"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(os.path.join(self.RESPONSES_PATH, "responses.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not save response artifact: {e}")
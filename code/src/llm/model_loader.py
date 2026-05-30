import logging
from transformers import pipeline
from src.config import LLM_MODEL_NAME, MAX_NEW_TOKENS

logger = logging.getLogger("SanskritRAG")


class LLMModel:
    """
    TinyLlama-1.1B-Chat-v1.0 loader via HuggingFace pipeline.

    Key behaviours:
    - Uses <|assistant|> token to extract only the answer (no prompt bleed)
    - do_sample=False for deterministic, faster inference on CPU
    - MAX_NEW_TOKENS pulled from config
    """

    ASSISTANT_TOKEN = "<|assistant|>"

    def __init__(self, model_name=LLM_MODEL_NAME):
        try:
            logger.info(f"Loading LLM: {model_name}")

            self.model_name = model_name

            self.generator = pipeline(
                task="text-generation",
                model=model_name,
                device=-1,              # CPU; set to 0 for GPU
                torch_dtype="auto",     # uses float32 on CPU automatically
            )

            logger.info("LLM loaded successfully.")

        except OSError as e:
            logger.error(f"Model not found: '{model_name}'. {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load LLM '{model_name}': {e}")
            raise RuntimeError(f"LLMModel init error: {e}") from e

    def generate_response(self, prompt, max_new_tokens=MAX_NEW_TOKENS):
        """
        Generate an answer from a TinyLlama-formatted prompt.
        Extracts only the text after <|assistant|> so the prompt
        never appears in the returned string.

        Args:
            prompt (str): Chat-formatted prompt ending with <|assistant|>
            max_new_tokens (int): Max tokens to generate
        Returns:
            str: Answer text only
        """

        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")

            logger.info(
                f"Generating response | "
                f"Prompt: {len(prompt)} chars | "
                f"max_new_tokens: {max_new_tokens}"
            )

            output = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # deterministic — faster + consistent on CPU
                repetition_penalty=1.1,     # prevents looping/repetition common in small models
                return_full_text=True,      # need full text to extract after <|assistant|>
            )

            full_text = output[0]["generated_text"]
            answer = self._extract_answer(full_text)

            if not answer.strip():
                logger.warning("LLM returned an empty answer.")
                answer = "The answer is not available in the provided document."

            logger.info(f"Answer extracted. Length: {len(answer)} chars.")
            return answer

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"generate_response error: {e}") from e

    def _extract_answer(self, full_text: str) -> str:
        """
        Extract only the assistant's reply from the full generated text.
        TinyLlama always produces <|assistant|> before its response.
        """
        if self.ASSISTANT_TOKEN in full_text:
            # Take everything after the LAST occurrence (in case prompt
            # contains an earlier <|assistant|> from few-shot examples)
            answer = full_text.rsplit(self.ASSISTANT_TOKEN, 1)[-1].strip()
            # Strip any trailing </s> end token
            answer = answer.replace("</s>", "").strip()
            return answer

        # Fallback: strip the prompt prefix if token not found
        logger.warning("<|assistant|> token not found in output. Returning raw output.")
        return full_text.strip()
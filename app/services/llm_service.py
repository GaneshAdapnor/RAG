"""
LLM service: answer generation using OpenAI GPT-4o-mini.

WHY GPT-4o-mini:
    - Strong instruction following: the model reliably respects the "answer
      only from context" constraint, making hallucination control effective.
    - Cost: ~$0.15 per 1M input tokens — significantly cheaper than GPT-4o
      ($2.50/M) while achieving comparable accuracy on focused QA tasks.
    - Speed: median latency ~1s for short answers, ~3s for long explanations.
    - Context window: 128K tokens — far more than we'll ever use (our context
      is ~600 tokens max with top_k=5 and 500-char chunks).

WHY NOT local LLMs (Ollama, llama.cpp):
    - Quality gap: 7B–13B local models perform significantly worse on multi-hop
      reasoning and faithful grounding compared to GPT-4o-mini.
    - Infrastructure: requires GPU or significant CPU/RAM. Not portable.
    - For a document QA API that must be accurate, API-based LLMs are the
      practical choice. Local models are viable for cost-sensitive,
      privacy-sensitive, or offline deployments.

HALLUCINATION PREVENTION (prompt engineering):
    The system prompt uses three key constraints:
    1. "Answer ONLY from the provided context" — explicit grounding instruction.
    2. "If the answer is not in the context, say 'I don't have enough
        information in the provided documents.'" — forces a refusal rather than
        a hallucinated guess.
    3. "Do not add information from your training data" — prevents the model
       from supplementing retrieved facts with memorized knowledge.

    These constraints are tested against the "needle in a haystack" failure:
    even if the LLM "knows" the answer from training, the prompt instruction
    overrides it when no supporting context is retrieved.
"""

import logging
import time
from typing import List

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from app.core.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Module-level OpenAI client — reused across requests (connection pooling)
_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """
    Return a singleton OpenAI client.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not configured.
    """
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment variables."
            )
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized (model: %s).", settings.LLM_MODEL_NAME)
    return _openai_client


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise document question-answering assistant.

RULES (non-negotiable):
1. Answer ONLY using information explicitly stated in the CONTEXT provided below.
2. If the context does not contain sufficient information to answer the question, respond with exactly: "I don't have enough information in the provided documents to answer this question."
3. Do NOT add information from your training data, prior knowledge, or assumptions.
4. Cite the source document name when you use information from it (e.g., "According to [filename]...").
5. Be concise and precise. Do not speculate or extrapolate.
6. If the question is ambiguous, answer the most likely interpretation based on the context.

CONTEXT:
{context}"""

USER_PROMPT_TEMPLATE = "Question: {query}"


def build_prompt(query: str, context: str) -> List[dict]:
    """
    Construct the OpenAI chat messages list.

    We use chat completion (not legacy completion) because:
    - System/user role separation gives the model clear behavioral guidance.
    - The model is trained to respect system prompt constraints in chat format.
    - All modern OpenAI models are optimized for chat completion.

    Args:
        query: The user's question.
        context: Formatted context string from retrieval_service.build_context().

    Returns:
        List of message dicts for the OpenAI chat completion API.
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context),
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(query=query),
        },
    ]


def generate_answer(
    query: str,
    context: str,
    chunks: List[RetrievedChunk],
) -> tuple[str, float]:
    """
    Generate a grounded answer from the LLM using the retrieved context.

    Args:
        query: Original user question.
        context: Formatted context string (output of build_context()).
        chunks: Retrieved chunks (used to detect empty context case).

    Returns:
        Tuple of (answer_text, generation_latency_ms).

    Raises:
        RuntimeError: For API authentication or server errors.
        ValueError: If context is empty (no documents indexed).
    """
    # Fast path: if no chunks were retrieved, skip the LLM call entirely
    if not chunks:
        logger.warning(
            "No chunks retrieved for query '%s...'. Returning no-context response.",
            query[:60],
        )
        return (
            "I don't have enough information in the provided documents to answer this question.",
            0.0,
        )

    if not context.strip():
        return (
            "I don't have enough information in the provided documents to answer this question.",
            0.0,
        )

    messages = build_prompt(query, context)

    client = get_openai_client()
    gen_start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=0.0,       # Deterministic — factual QA benefits from no sampling
            max_tokens=1024,       # Sufficient for detailed answers; cap prevents runaway cost
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        gen_latency_ms = (time.perf_counter() - gen_start) * 1000

        answer = response.choices[0].message.content or ""
        answer = answer.strip()

        logger.info(
            "LLM answer generated: model=%s, prompt_tokens=%d, completion_tokens=%d, "
            "latency=%.1fms.",
            settings.LLM_MODEL_NAME,
            response.usage.prompt_tokens if response.usage else -1,
            response.usage.completion_tokens if response.usage else -1,
            gen_latency_ms,
        )

        return answer, gen_latency_ms

    except RateLimitError as exc:
        logger.error("OpenAI rate limit exceeded: %s", exc)
        raise RuntimeError(
            "OpenAI API rate limit exceeded. Please wait and try again."
        ) from exc

    except APIConnectionError as exc:
        logger.error("OpenAI connection error: %s", exc)
        raise RuntimeError(
            "Could not connect to the OpenAI API. Check network connectivity."
        ) from exc

    except APIError as exc:
        logger.error("OpenAI API error (status %s): %s", exc.status_code, exc)
        raise RuntimeError(f"OpenAI API error: {exc.message}") from exc

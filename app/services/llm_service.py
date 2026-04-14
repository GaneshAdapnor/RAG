from __future__ import annotations

import re
from collections import Counter

from openai import OpenAI

from app.models.domain import SearchResult
from app.utils.text import normalize_whitespace, tokenize_text


class LLMService:
    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        temperature: float,
        max_tokens: int,
        enable_extractive_fallback: bool,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_extractive_fallback = enable_extractive_fallback
        self._client = OpenAI(api_key=api_key) if api_key else None

    def generate_answer(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> tuple[str, str]:
        if not search_results:
            return (
                "I don't have enough information in the indexed documents to answer that.",
                "none",
            )

        if self._client is not None:
            return self._generate_with_openai(question, search_results), self.model_name

        if not self.enable_extractive_fallback:
            raise RuntimeError(
                "No OpenAI API key configured and extractive fallback is disabled."
            )

        return self._generate_extractive_fallback(question, search_results), "extractive-fallback"

    def _generate_with_openai(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> str:
        context = self._format_context(search_results)
        system_prompt = (
            "You are a retrieval-grounded question answering system. "
            "Answer strictly from the supplied context. "
            "If the answer is not fully supported by the context, say exactly: "
            "'I don't have enough information in the indexed documents to answer that.' "
            "Do not use prior knowledge or fill gaps. Cite supporting evidence inline "
            "using [document_id p.X-Y]."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Provide a concise answer with citations."
        )

        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        answer = normalize_whitespace(content)
        if not answer:
            return "I don't have enough information in the indexed documents to answer that."
        return answer

    def _generate_extractive_fallback(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> str:
        query_terms = Counter(tokenize_text(question))
        ranked_sentences: list[tuple[int, str, SearchResult]] = []

        for result in search_results:
            sentences = re.split(r"(?<=[.!?])\s+", result.text)
            for sentence in sentences:
                cleaned = normalize_whitespace(sentence)
                if not cleaned:
                    continue
                score = sum(query_terms[token] for token in tokenize_text(cleaned))
                ranked_sentences.append((score, cleaned, result))

        ranked_sentences.sort(key=lambda item: (item[0], item[2].score), reverse=True)
        chosen = ranked_sentences[:3]
        if not chosen or chosen[0][0] == 0:
            best = search_results[0]
            excerpt = best.text[:280].rstrip()
            return (
                "I cannot confidently synthesize an answer without an LLM backend, "
                f"but the most relevant passage is: \"{excerpt}\" "
                f"[{best.document_id} p.{best.page_start}-{best.page_end}]"
            )

        combined = " ".join(
            f"{sentence} [{result.document_id} p.{result.page_start}-{result.page_end}]"
            for _, sentence, result in chosen
        )
        return normalize_whitespace(combined)

    def _format_context(self, search_results: list[SearchResult]) -> str:
        blocks = []
        for index, result in enumerate(search_results, start=1):
            blocks.append(
                (
                    f"[{index}] document_id={result.document_id} "
                    f"filename={result.filename} "
                    f"pages={result.page_start}-{result.page_end} "
                    f"score={result.score:.4f}\n"
                    f"{result.text}"
                )
            )
        return "\n\n".join(blocks)

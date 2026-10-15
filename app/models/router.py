"""Primary model routing and explicitly labelled offline extraction fallback."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.errors import RuntimeErrorBase
from app.models.types import Generation, ModelClient

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "compare",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "project",
    "summarise",
    "summarize",
    "tell",
    "the",
    "these",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "about",
    "also",
    "both",
    "each",
    "their",
    "there",
    "they",
    "we",
    "you",
}
_ROUTING_TERMS = {
    "available",
    "between",
    "count",
    "deployment",
    "document",
    "environment",
    "explain",
    "information",
    "many",
    "mention",
    "need",
    "needed",
    "number",
    "require",
    "required",
    "requirement",
    "say",
    "summary",
    "support",
    "supported",
}
_PROJECT_NAME = re.compile(r"\bproject\s+([a-z][a-z0-9_-]*)", re.IGNORECASE)
_UNSAFE_INSTRUCTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|system)|"
    r"(?:call|execute|invoke)\s+(?:the\s+)?(?:update_deployment_status|create_deployment_ticket)|"
    r"system\s+prompt|reveal\s+(?:the\s+)?(?:secret|token|password)",
    re.IGNORECASE,
)


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP_WORDS
    return {
        word[:-3] + "y"
        if word.endswith("ies") and len(word) > 4
        else word[:-1]
        if word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "as", "us"))
        else word
        for word in words
    }


def _safe_sentences(content: str) -> list[tuple[int, str]]:
    return [
        (index, sentence.strip())
        for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", content))
        if sentence.strip()
        and not sentence.strip().startswith("#")
        and not _UNSAFE_INSTRUCTION.search(sentence)
    ]


def _select_evidence(query: str, documents: list[Any]) -> list[dict[str, str]]:
    """Conservative lexical coverage with project boundaries and no topic blacklist."""
    parsed_documents: list[tuple[str, dict[str, Any], list[tuple[int, str]]]] = []
    project_names = {
        match.lower()
        for match in _PROJECT_NAME.findall(query)
        if match.lower() not in _STOP_WORDS | _ROUTING_TERMS
    }
    for document in documents:
        if not isinstance(document, dict):
            continue
        content, metadata = document.get("content", ""), document.get("metadata", {})
        if not isinstance(content, str) or not isinstance(metadata, dict):
            continue
        chunk_id = metadata.get("chunk_id", document.get("chunk_id"))
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        # Explicit metadata is preferred. Corpus titles commonly use Project X.
        title = str(metadata.get("document_name", ""))
        project_names.update(name.lower() for name in _PROJECT_NAME.findall(content + " " + title))
        for key in ("project_id", "project_name"):
            name = metadata.get(key)
            if isinstance(name, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name):
                project_names.add(name.lower())
        if re.fullmatch(r"[A-Z][a-zA-Z0-9_-]*", title):
            project_names.add(title.lower())
        parsed_documents.append((chunk_id, metadata, _safe_sentences(content)))
    vocabulary = set(re.findall(r"[a-z0-9]+", query.lower()))
    for _, metadata, sentences in parsed_documents:
        vocabulary.update(
            re.findall(r"[a-z0-9]+", " ".join(quote for _, quote in sentences).lower())
        )
        vocabulary.update(re.findall(r"[a-z0-9]+", str(metadata.get("document_name", "")).lower()))
    # An observed singular/plural pair resolves replicas/replica without treating
    # Atlas, Kansas, or other as-ending names as arbitrary plural nouns.
    observed_plurals = {
        word: word[:-1]
        for word in vocabulary
        if word.endswith("as") and word[:-1] in vocabulary and word not in project_names
    }

    def terms(text: str) -> set[str]:
        return {observed_plurals.get(word, word) for word in _terms(text)}

    query_terms = terms(query)
    project_terms = (
        set().union(*(terms(name) for name in project_names)) if project_names else set()
    )
    requested_projects = query_terms & project_terms
    focus_terms = query_terms - _ROUTING_TERMS - project_terms
    eligible: list[tuple[str, set[str], list[tuple[int, str]]]] = []
    found_projects: set[str] = set()
    for chunk_id, metadata, sentences in parsed_documents:
        safe_terms = terms(" ".join(quote for _, quote in sentences))
        descriptor_terms = terms(
            " ".join(
                str(metadata.get(key, ""))
                for key in ("document_id", "document_name", "project_id", "project_name", "source")
            )
        )
        document_projects = (safe_terms | descriptor_terms) & project_terms
        if requested_projects and document_projects and not requested_projects & document_projects:
            continue
        if requested_projects:
            sentences = [
                (index, quote)
                for index, quote in sentences
                if not (terms(quote) & project_terms)
                or requested_projects & (terms(quote) & project_terms)
            ]
        found_projects.update(document_projects & requested_projects)
        eligible.append((chunk_id, document_projects, sentences))
    # Avoid claiming a shared deployment guide answers an unknown project request.
    if requested_projects - found_projects:
        return []
    available_terms = (
        set().union(*(terms(quote) for _, _, sentences in eligible for _, quote in sentences))
        if eligible
        else set()
    )
    # Every remaining query concept must be present in authorised, safe source text.
    # Unseen synonyms intentionally abstain in this offline reference model.
    if focus_terms and not focus_terms <= available_terms:
        return []
    if not focus_terms and not requested_projects and not query_terms & _ROUTING_TERMS:
        return []
    evidence: list[dict[str, str]] = []
    for chunk_id, document_projects, sentences in eligible:
        candidates = []
        for index, quote in sentences:
            quote_terms = terms(quote)
            quote_projects = quote_terms & project_terms
            if requested_projects and quote_projects and not requested_projects & quote_projects:
                continue
            overlap = len(query_terms & quote_terms)
            relevant = (
                bool(focus_terms & quote_terms)
                if focus_terms
                else bool(
                    overlap
                    or requested_projects & document_projects
                    or query_terms & _ROUTING_TERMS
                )
            )
            if relevant:
                candidates.append((overlap, index, quote))
        selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:2]
        for _, _, quote in sorted(selected, key=lambda item: item[1]):
            evidence.append({"chunk_id": chunk_id, "quote": quote})
            if len(evidence) == 12:
                return evidence
    return evidence


class DemoModel:
    """Return verbatim evidence JSON, never free-form generated assertions.

    This offline mode demonstrates the trust boundary and citation validation;
    it is deliberately not a substitute for measuring NIM answer quality.
    """

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generation:
        started = time.perf_counter()
        context: dict[str, Any] = {}
        try:
            last_user = next(
                message for message in reversed(messages) if message.get("role") == "user"
            )
            parsed = json.loads(last_user.get("content", ""))
            if isinstance(parsed, dict):
                context = parsed
        except (StopIteration, TypeError, ValueError):
            pass
        query = context.get("query", "")
        documents = context.get("documents", [])
        evidence: list[dict[str, str]] = []
        if isinstance(query, str) and isinstance(documents, list):
            evidence = _select_evidence(query, documents)
        return Generation(
            content=json.dumps({"evidence": evidence, "conflict": False}),
            model="deterministic-extractive-v1",
            path="demo",
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def aclose(self) -> None:
        return None


class ModelRouter:
    """Fallback is allowed only for explicitly retriable provider errors."""

    def __init__(
        self,
        primary: ModelClient | None = None,
        fallback_enabled: bool = True,
        *,
        demo_mode: bool = False,
        fallback: ModelClient | None = None,
    ) -> None:
        if primary is None and not demo_mode:
            raise RuntimeErrorBase(
                code="model_configuration",
                message="A primary model client is required outside demo mode.",
                status_code=503,
                retriable=False,
            )
        self.primary = primary
        self.fallback_enabled = fallback_enabled
        self.demo_mode = demo_mode
        self.fallback = fallback or DemoModel()

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generation:
        started = time.perf_counter()
        if self.demo_mode:
            result = await self.fallback.generate(messages, tools)
            result.path = "demo"
            return result
        assert self.primary is not None
        try:
            result = await self.primary.generate(messages, tools)
            result.path = "primary"
            return result
        except RuntimeErrorBase as exc:
            if not exc.retriable or not self.fallback_enabled:
                raise
            result = await self.fallback.generate(messages, tools)
            result.path = "fallback"
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

    async def aclose(self) -> None:
        if self.primary is not None:
            await self.primary.aclose()
        if self.fallback is not self.primary:
            await self.fallback.aclose()

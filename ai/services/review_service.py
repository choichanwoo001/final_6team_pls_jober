import asyncio
import time
from typing import Any, Dict, List, Tuple

from services.cache_service import CacheService
from services.chromadb_service import ChromaDBService
from services.huggingface_service import HuggingFaceService


class ReviewService:
    PRECHECK_TTL_SECONDS = 600
    SEARCH_TTL_SECONDS = 600
    SENTIMENT_TTL_SECONDS = 1800
    QA_TTL_SECONDS = 1800

    def __init__(
        self,
        chromadb_service: ChromaDBService,
        huggingface_service: HuggingFaceService,
        cache_service: CacheService,
    ):
        self.chromadb_service = chromadb_service
        self.huggingface_service = huggingface_service
        self.cache_service = cache_service

    async def precheck(self, content: str, question: str, n_results: int = 5) -> Dict[str, Any]:
        total_start = time.perf_counter()
        precheck_key = self.cache_service.build_key("precheck", content, question, n_results)
        cached = await self.cache_service.get_json(precheck_key)
        if cached is not None:
            cached.setdefault("metrics", {})
            cached["metrics"]["cacheHit"] = True
            cached["metrics"]["totalMs"] = self._elapsed_ms(total_start)
            return cached

        search_task = self._timed(
            self._cached_search(content, n_results),
            "searchMs",
        )
        sentiment_task = self._timed(
            self._cached_sentiment(content),
            "sentimentMs",
        )
        qa_task = self._timed(
            self._cached_qa(question, content),
            "qaMs",
        )

        (search_result, search_ms), (sentiment_result, sentiment_ms), (qa_result, qa_ms) = await asyncio.gather(
            search_task,
            sentiment_task,
            qa_task,
        )

        score, risk, reasons, suggestions = self._score_review(
            content,
            search_result,
            sentiment_result,
            qa_result,
        )

        response = {
            "score": score,
            "approvalRisk": risk,
            "reasons": reasons,
            "suggestions": suggestions,
            "similarDocuments": self._similar_documents(search_result),
            "sentiment": sentiment_result.get("sentiment", {}),
            "qa": qa_result,
            "metrics": {
                "totalMs": self._elapsed_ms(total_start),
                "searchMs": search_ms,
                "sentimentMs": sentiment_ms,
                "qaMs": qa_ms,
                "cacheHit": False,
            },
        }
        await self.cache_service.set_json(precheck_key, response, self.PRECHECK_TTL_SECONDS)
        return response

    async def _cached_search(self, content: str, n_results: int) -> Dict[str, Any]:
        key = self.cache_service.build_key("search", content, n_results)
        cached = await self.cache_service.get_json(key)
        if cached is not None:
            return cached
        try:
            result = await self.chromadb_service.search_documents(content, n_results)
        except Exception:
            result = {"query": content, "results": [], "metadatas": [], "distances": []}
        await self.cache_service.set_json(key, result, self.SEARCH_TTL_SECONDS)
        return result

    async def _cached_sentiment(self, content: str) -> Dict[str, Any]:
        key = self.cache_service.build_key("sentiment", content)
        cached = await self.cache_service.get_json(key)
        if cached is not None:
            return cached
        try:
            result = await self.huggingface_service.analyze_sentiment(content)
        except Exception:
            result = {"sentiment": {"label": "UNKNOWN", "score": 0.0}}
        await self.cache_service.set_json(key, result, self.SENTIMENT_TTL_SECONDS)
        return result

    async def _cached_qa(self, question: str, content: str) -> Dict[str, Any]:
        key = self.cache_service.build_key("qa", question, content)
        cached = await self.cache_service.get_json(key)
        if cached is not None:
            return cached
        try:
            result = await self.huggingface_service.answer_question(question, content)
        except Exception:
            result = {"answer": "", "score": 0.0, "start": 0, "end": 0}
        await self.cache_service.set_json(key, result, self.QA_TTL_SECONDS)
        return result

    async def _timed(self, task, metric_name: str) -> Tuple[Dict[str, Any], int]:
        start = time.perf_counter()
        result = await task
        return result, self._elapsed_ms(start)

    def _score_review(
        self,
        content: str,
        search_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        qa_result: Dict[str, Any],
    ) -> Tuple[int, str, List[str], List[str]]:
        score = 70
        reasons: List[str] = []
        suggestions: List[str] = []

        qa_score = float(qa_result.get("score") or 0)
        if qa_score >= 0.75:
            score += 15
            reasons.append("QA score is above the approval threshold.")
        elif qa_score < 0.5:
            score -= 20
            reasons.append("QA score is low, so the content may not satisfy the requirement.")
            suggestions.append("Add clearer details that directly answer the required condition.")

        distances = search_result.get("distances") or []
        if distances and min(distances) <= 0.35:
            score += 10
            reasons.append("A highly similar reference document was found.")
        elif not distances:
            suggestions.append("Add reference documents to improve similarity-based validation.")

        sentiment = sentiment_result.get("sentiment") or {}
        label = str(sentiment.get("label", "")).upper()
        sentiment_score = float(sentiment.get("score") or 0)
        if label in {"NEGATIVE", "LABEL_0"} and sentiment_score >= 0.7:
            score -= 15
            reasons.append("Strong negative sentiment was detected.")
            suggestions.append("Rewrite negative or unclear expressions in a neutral tone.")
        else:
            reasons.append("No strong negative sentiment was detected.")

        if len(content.strip()) < 30:
            score -= 10
            reasons.append("Content is too short for stable validation.")
            suggestions.append("Write at least two or three concrete sentences before submitting.")

        score = max(0, min(100, score))
        if score >= 80:
            risk = "LOW"
        elif score >= 50:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        if not suggestions:
            suggestions.append("Keep the current structure and make key requirements explicit.")

        return score, risk, reasons, suggestions

    def _similar_documents(self, search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = search_result.get("results") or []
        metadatas = search_result.get("metadatas") or []
        distances = search_result.get("distances") or []
        similar_documents = []
        for index, document in enumerate(documents):
            similar_documents.append(
                {
                    "content": document,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return similar_documents

    def _elapsed_ms(self, start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

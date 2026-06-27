from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from services.cache_service import CacheService
from services.chromadb_service import ChromaDBService
from services.huggingface_service import HuggingFaceService
from services.openai_service import OpenAIService
from services.review_service import ReviewService

router = APIRouter(prefix="/ai", tags=["AI Services"])

openai_service = OpenAIService()
chromadb_service = ChromaDBService()
huggingface_service = HuggingFaceService()
cache_service = CacheService()
review_service = ReviewService(chromadb_service, huggingface_service, cache_service)


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-3.5-turbo"


class ChatResponse(BaseModel):
    response: str
    model: str


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5


class TextGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt2"
    max_length: Optional[int] = 100


class SentimentRequest(BaseModel):
    text: str
    model: Optional[str] = "cardiffnlp/twitter-roberta-base-sentiment"


class QuestionAnswerRequest(BaseModel):
    question: str
    context: str
    model: Optional[str] = "deepset/roberta-base-squad2"


class ReviewPrecheckRequest(BaseModel):
    content: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    n_results: int = Field(default=5, ge=1, le=20)


async def warmup_models():
    await huggingface_service.warmup_common_models()


@router.post("/review/precheck")
async def precheck_review(request: ReviewPrecheckRequest):
    try:
        return await review_service.precheck(
            request.content,
            request.question,
            request.n_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/openai/chat", response_model=ChatResponse)
async def openai_chat(request: ChatRequest):
    try:
        messages = [{"role": "user", "content": request.message}]
        response = await openai_service.chat_completion(messages, request.model)
        return ChatResponse(response=response, model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/openai/embeddings")
async def openai_embeddings(text: str):
    try:
        embeddings = await openai_service.embeddings(text)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chromadb/documents")
async def add_documents(request: DocumentRequest):
    try:
        metadata = request.metadata or {"source": "user_input"}
        result = await chromadb_service.add_documents([request.content], [metadata])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chromadb/search")
async def search_documents(request: SearchRequest):
    try:
        result = await chromadb_service.search_documents(request.query, request.n_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chromadb/info")
async def get_collection_info():
    try:
        result = await chromadb_service.get_collection_info()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chromadb/documents/{document_id}")
async def get_document(document_id: str):
    try:
        result = await chromadb_service.get_document_by_id(document_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/huggingface/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        result = await huggingface_service.generate_text(
            request.prompt,
            request.model,
            request.max_length,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/huggingface/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = await huggingface_service.analyze_sentiment(request.text, request.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/huggingface/embeddings")
async def get_embeddings(texts: List[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        result = await huggingface_service.get_embeddings(texts, model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/huggingface/qa")
async def question_answering(request: QuestionAnswerRequest):
    try:
        result = await huggingface_service.answer_question(
            request.question,
            request.context,
            request.model,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface/models")
async def get_available_models():
    try:
        result = await huggingface_service.get_available_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

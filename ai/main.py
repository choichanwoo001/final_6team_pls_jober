from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from routers import ai_routes
import asyncio

load_dotenv()

app = FastAPI(
    title="AI Service API",
    description="FastAPI + ChromaDB + OpenAI + Hugging Face AI service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_routes.router)


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-3.5-turbo"


class ChatResponse(BaseModel):
    response: str
    model: str


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ai_routes.warmup_models())


@app.get("/")
async def root():
    return {"message": "AI Service is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = f"AI response: {request.message}"
        return ChatResponse(response=response, model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_document(request: DocumentRequest):
    try:
        return {"message": "Document added successfully", "content": request.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: SearchRequest):
    try:
        results = [f"Search result {i + 1}: {request.query}" for i in range(request.n_results)]
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

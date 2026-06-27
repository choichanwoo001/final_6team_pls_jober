import asyncio
import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

logger = logging.getLogger(__name__)


class HuggingFaceService:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.pipelines = {}

    async def load_text_generation_model(self, model_name: str = "gpt2"):
        try:
            if model_name not in self.pipelines:
                logger.info("Loading text generation model: %s", model_name)
                self.pipelines[model_name] = await asyncio.to_thread(
                    pipeline,
                    "text-generation",
                    model=model_name,
                    device=self.device,
                )
            else:
                logger.info("Model cache hit: %s", model_name)
            return {"message": f"{model_name} model is ready"}
        except Exception as e:
            raise Exception(f"Model load failed: {str(e)}")

    async def generate_text(self, prompt: str, model_name: str = "gpt2", max_length: int = 100):
        try:
            if model_name not in self.pipelines:
                await self.load_text_generation_model(model_name)

            result = await asyncio.to_thread(
                self.pipelines[model_name],
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
            )
            return {"generated_text": result[0]["generated_text"]}
        except Exception as e:
            raise Exception(f"Text generation failed: {str(e)}")

    async def load_sentiment_analysis_model(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ):
        try:
            if model_name not in self.pipelines:
                logger.info("Loading sentiment model: %s", model_name)
                self.pipelines[model_name] = await asyncio.to_thread(
                    pipeline,
                    "sentiment-analysis",
                    model=model_name,
                    device=self.device,
                )
            else:
                logger.info("Model cache hit: %s", model_name)
            return {"message": f"{model_name} model is ready"}
        except Exception as e:
            raise Exception(f"Model load failed: {str(e)}")

    async def analyze_sentiment(
        self,
        text: str,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ):
        try:
            if model_name not in self.pipelines:
                await self.load_sentiment_analysis_model(model_name)

            result = await asyncio.to_thread(self.pipelines[model_name], text)
            return {"sentiment": result[0]}
        except Exception as e:
            raise Exception(f"Sentiment analysis failed: {str(e)}")

    async def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            if model_name not in self.models:
                logger.info("Loading embedding model: %s", model_name)
                self.models[model_name] = await asyncio.to_thread(
                    SentenceTransformer,
                    model_name,
                    device=self.embedding_device,
                )
            else:
                logger.info("Model cache hit: %s", model_name)
            return {"message": f"{model_name} model is ready"}
        except Exception as e:
            raise Exception(f"Model load failed: {str(e)}")

    async def get_embeddings(
        self,
        texts: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        try:
            if model_name not in self.models:
                await self.load_embedding_model(model_name)

            embeddings = await asyncio.to_thread(self.models[model_name].encode, texts)
            return {"embeddings": embeddings.tolist()}
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")

    async def load_question_answering_model(self, model_name: str = "deepset/roberta-base-squad2"):
        try:
            if model_name not in self.pipelines:
                logger.info("Loading QA model: %s", model_name)
                self.pipelines[model_name] = await asyncio.to_thread(
                    pipeline,
                    "question-answering",
                    model=model_name,
                    device=self.device,
                )
            else:
                logger.info("Model cache hit: %s", model_name)
            return {"message": f"{model_name} model is ready"}
        except Exception as e:
            raise Exception(f"Model load failed: {str(e)}")

    async def answer_question(
        self,
        question: str,
        context: str,
        model_name: str = "deepset/roberta-base-squad2",
    ):
        try:
            if model_name not in self.pipelines:
                await self.load_question_answering_model(model_name)

            result = await asyncio.to_thread(
                self.pipelines[model_name],
                question=question,
                context=context,
            )
            return {
                "answer": result["answer"],
                "score": result["score"],
                "start": result["start"],
                "end": result["end"],
            }
        except Exception as e:
            raise Exception(f"Question answering failed: {str(e)}")

    async def warmup_common_models(self):
        for loader in (
            self.load_sentiment_analysis_model,
            self.load_question_answering_model,
        ):
            try:
                await loader()
            except Exception as e:
                logger.warning("Model pre-warming skipped: %s", e)

    async def get_available_models(self):
        return {
            "text_generation": ["gpt2", "gpt2-medium", "gpt2-large"],
            "sentiment_analysis": [
                "cardiffnlp/twitter-roberta-base-sentiment",
                "distilbert-base-uncased-finetuned-sst-2-english",
            ],
            "embedding": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            "question_answering": [
                "deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
            ],
        }

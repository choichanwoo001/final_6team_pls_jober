import os
from typing import Dict, List, Optional

from openai import OpenAI


class OpenAIService:
    def __init__(self):
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY is not configured")
            self._client = OpenAI(api_key=api_key)
        return self._client

    async def chat_completion(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    async def text_completion(self, prompt: str, model: str = "text-davinci-003") -> str:
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7,
            )
            return response.choices[0].text
        except Exception as e:
            raise Exception(f"OpenAI completion failed: {str(e)}")

    async def embeddings(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding failed: {str(e)}")

    async def batch_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
    ) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise Exception(f"OpenAI batch embedding failed: {str(e)}")

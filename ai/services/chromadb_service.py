import asyncio
import uuid
from typing import Any, Dict, List, Optional

import chromadb


class ChromaDBService:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(name=self.collection_name)

    async def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        try:
            ids = ids or [str(uuid.uuid4()) for _ in documents]
            metadatas = metadatas or [{"source": "user_input"} for _ in documents]

            await asyncio.to_thread(
                self.collection.add,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            return {"message": f"{len(documents)} documents added", "ids": ids}
        except Exception as e:
            raise Exception(f"Document add failed: {str(e)}")

    async def search_documents(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ):
        try:
            count = await asyncio.to_thread(self.collection.count)
            if count == 0:
                return {"query": query, "results": [], "metadatas": [], "distances": []}

            result_count = min(n_results, count)
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=result_count,
                where=where,
            )
            return {
                "query": query,
                "results": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }
        except Exception as e:
            raise Exception(f"Document search failed: {str(e)}")

    async def get_document_by_id(self, document_id: str):
        try:
            results = await asyncio.to_thread(self.collection.get, ids=[document_id])
            if results["documents"]:
                return {
                    "id": document_id,
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                }
            return None
        except Exception as e:
            raise Exception(f"Document lookup failed: {str(e)}")

    async def update_document(
        self,
        document_id: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            await asyncio.to_thread(
                self.collection.update,
                ids=[document_id],
                documents=[document],
                metadatas=[metadata] if metadata else None,
            )
            return {"message": "Document updated", "id": document_id}
        except Exception as e:
            raise Exception(f"Document update failed: {str(e)}")

    async def delete_document(self, document_id: str):
        try:
            await asyncio.to_thread(self.collection.delete, ids=[document_id])
            return {"message": "Document deleted", "id": document_id}
        except Exception as e:
            raise Exception(f"Document delete failed: {str(e)}")

    async def get_collection_info(self):
        try:
            count = await asyncio.to_thread(self.collection.count)
            return {
                "collection_name": self.collection_name,
                "document_count": count,
            }
        except Exception as e:
            raise Exception(f"Collection info lookup failed: {str(e)}")

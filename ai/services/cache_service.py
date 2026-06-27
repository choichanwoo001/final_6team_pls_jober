import hashlib
import json
import logging
import os
from typing import Any, Optional

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency fallback
    redis = None

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        self.enabled = redis is not None
        self.client = None
        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    db=int(os.getenv("REDIS_DB", "0")),
                    socket_connect_timeout=1,
                    socket_timeout=1,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning("Redis cache disabled: %s", e)
                self.enabled = False

    def build_key(self, prefix: str, *parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    async def get_json(self, key: str) -> Optional[Any]:
        if not self.enabled or self.client is None:
            return None
        try:
            value = await self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning("Redis cache read skipped: %s", e)
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            await self.client.setex(
                key,
                ttl_seconds,
                json.dumps(value, ensure_ascii=False, default=str),
            )
        except Exception as e:
            logger.warning("Redis cache write skipped: %s", e)

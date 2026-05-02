"""Qwen3-VL Image-Aware Context Compressor — Open WebUI Filter Function.

Paste this entire file into Admin Panel → Functions → New Function.
Source of truth lives in the repo at vast-templates/qwen3-vl-8b/functions/.
"""

import asyncio
import os
import time
from typing import Iterator, Optional

import aiosqlite

VERSION = "0.1.0-dev"


class CaptionCache:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS captions (
        img_hash    TEXT PRIMARY KEY,
        caption     TEXT NOT NULL,
        model       TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        bytes_size  INTEGER,
        user_id     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_created ON captions(created_at);
    """

    def __init__(self, path: str):
        self.path = path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            async with aiosqlite.connect(self.path) as db:
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA synchronous = NORMAL")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.executescript(self.SCHEMA)
                await db.commit()
            self._initialized = True

    async def get(self, h: str) -> Optional[str]:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT caption FROM captions WHERE img_hash = ?", (h,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def put(
        self,
        h: str,
        caption: str,
        model: str,
        bytes_size: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (h, caption, model, now_ms, bytes_size, user_id),
            )
            await db.commit()

    async def get_many(self, hashes: list[str]) -> dict[str, str]:
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        async with aiosqlite.connect(self.path) as db:
            sql = f"SELECT img_hash, caption FROM captions WHERE img_hash IN ({placeholders})"
            async with db.execute(sql, hashes) as cur:
                return {h: c async for h, c in cur}

    async def put_many(self, items: list[tuple[str, str, str, Optional[int], Optional[str]]]) -> None:
        """items: (hash, caption, model, bytes_size, user_id)."""
        if not items:
            return
        now_ms = int(time.time() * 1000)
        rows = [(h, c, m, now_ms, sz, uid) for h, c, m, sz, uid in items]
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()


def has_images(msg: dict) -> bool:
    """Check if a message contains any image_url parts."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(p.get("type") == "image_url" for p in content)


def iter_image_parts(msgs: list[dict]) -> Iterator[tuple[int, int, str]]:
    """Yield (msg_idx, content_idx, url) for every image_url part."""
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    yield i, j, url


def find_latest_image_turn(msgs: list[dict]) -> Optional[int]:
    """Return the index of the latest user turn with images, or None."""
    latest: Optional[int] = None
    for i, msg in enumerate(msgs):
        if msg.get("role") == "user" and has_images(msg):
            latest = i
    return latest


def text_of(msg: dict) -> str:
    """Extract text content from a message (string or multimodal)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
    return ""

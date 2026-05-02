"""Qwen3-VL Image-Aware Context Compressor — Open WebUI Filter Function.

Paste this entire file into Admin Panel → Functions → New Function.
Source of truth lives in the repo at vast-templates/qwen3-vl-8b/functions/.
"""

import asyncio
import os
import time
from typing import Optional

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

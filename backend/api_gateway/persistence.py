"""Persistence layer for API Gateway services.

Provides PostgreSQL-backed storage for sessions, pipelines, and research jobs
when DATABASE_URL is configured. Falls back to in-memory storage otherwise.

Uses asyncpg for async database operations with connection pooling.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any


class DatabasePool:
    """Manages an asyncpg connection pool.

    Lazily initializes the pool on first use. Falls back gracefully
    if asyncpg is not available or DATABASE_URL is not set.
    """

    _pool = None
    _initialized = False

    @classmethod
    async def get_pool(cls):
        """Get or create the connection pool."""
        if cls._initialized:
            return cls._pool

        cls._initialized = True
        database_url = os.environ.get("DATABASE_URL", "")

        if not database_url:
            cls._pool = None
            return None

        try:
            import asyncpg
            cls._pool = await asyncpg.create_pool(
                database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            return cls._pool
        except Exception:
            cls._pool = None
            return None

    @classmethod
    async def close(cls):
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            cls._initialized = False


class SessionRepository:
    """PostgreSQL-backed session storage.

    Falls back to in-memory dict if no database is available.
    """

    def __init__(self):
        self._memory: dict[str, dict] = {}

    async def create(self, session_id: str, tenant_id: str, retention_days: int, created_at: datetime) -> None:
        """Persist a new session."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO sessions (session_id, tenant_id, retention_days, created_at, state, memory)
                       VALUES ($1, $2, $3, $4, 'active', '{}')""",
                    session_id, tenant_id, retention_days, created_at,
                )
        else:
            self._memory[session_id] = {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "retention_days": retention_days,
                "created_at": created_at,
                "state": "active",
                "memory": {},
            }

    async def get(self, session_id: str, tenant_id: str) -> dict | None:
        """Get a session by ID with tenant isolation."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT session_id, tenant_id, retention_days, created_at, state, memory
                       FROM sessions WHERE session_id = $1 AND tenant_id = $2 AND state = 'active'""",
                    session_id, tenant_id,
                )
                if row:
                    return dict(row)
                return None
        else:
            session = self._memory.get(session_id)
            if session and session["tenant_id"] == tenant_id and session["state"] == "active":
                return session
            return None

    async def delete(self, session_id: str, tenant_id: str) -> bool:
        """Mark a session as deleted."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """UPDATE sessions SET state = 'deleted' WHERE session_id = $1 AND tenant_id = $2""",
                    session_id, tenant_id,
                )
                return "UPDATE 1" in result
        else:
            session = self._memory.get(session_id)
            if session and session["tenant_id"] == tenant_id:
                session["state"] = "deleted"
                return True
            return False


class PipelineRepository:
    """PostgreSQL-backed pipeline storage.

    Falls back to in-memory dict if no database is available.
    """

    def __init__(self):
        self._memory: dict[str, dict] = {}

    async def create(self, pipeline_id: str, tenant_id: str, name: str, steps: list[dict], created_at: datetime) -> None:
        """Persist a new pipeline."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO pipelines (pipeline_id, tenant_id, name, steps, created_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    pipeline_id, tenant_id, name, json.dumps(steps), created_at,
                )
        else:
            self._memory[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "tenant_id": tenant_id,
                "name": name,
                "steps": steps,
                "created_at": created_at,
            }

    async def get(self, pipeline_id: str, tenant_id: str) -> dict | None:
        """Get a pipeline by ID with tenant isolation."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT pipeline_id, tenant_id, name, steps, created_at
                       FROM pipelines WHERE pipeline_id = $1 AND tenant_id = $2""",
                    pipeline_id, tenant_id,
                )
                if row:
                    result = dict(row)
                    result["steps"] = json.loads(result["steps"])
                    return result
                return None
        else:
            pipeline = self._memory.get(pipeline_id)
            if pipeline and pipeline["tenant_id"] == tenant_id:
                return pipeline
            return None

    async def delete(self, pipeline_id: str, tenant_id: str) -> bool:
        """Delete a pipeline."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """DELETE FROM pipelines WHERE pipeline_id = $1 AND tenant_id = $2""",
                    pipeline_id, tenant_id,
                )
                return "DELETE 1" in result
        else:
            pipeline = self._memory.get(pipeline_id)
            if pipeline and pipeline["tenant_id"] == tenant_id:
                del self._memory[pipeline_id]
                return True
            return False


class ResearchJobRepository:
    """PostgreSQL-backed research job storage.

    Falls back to in-memory dict if no database is available.
    """

    def __init__(self):
        self._memory: dict[str, dict] = {}

    async def create(self, job_id: str, tenant_id: str, research_goal: str, state: str, created_at: datetime) -> None:
        """Persist a new research job."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO research_jobs (job_id, tenant_id, research_goal, state, created_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    job_id, tenant_id, research_goal, state, created_at,
                )
        else:
            self._memory[job_id] = {
                "job_id": job_id,
                "tenant_id": tenant_id,
                "research_goal": research_goal,
                "state": state,
                "created_at": created_at,
            }

    async def get(self, job_id: str, tenant_id: str) -> dict | None:
        """Get a research job by ID with tenant isolation."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT job_id, tenant_id, research_goal, state, created_at
                       FROM research_jobs WHERE job_id = $1 AND tenant_id = $2""",
                    job_id, tenant_id,
                )
                if row:
                    return dict(row)
                return None
        else:
            job = self._memory.get(job_id)
            if job and job["tenant_id"] == tenant_id:
                return job
            return None

    async def update_state(self, job_id: str, state: str) -> None:
        """Update job state."""
        pool = await DatabasePool.get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    """UPDATE research_jobs SET state = $1 WHERE job_id = $2""",
                    state, job_id,
                )
        else:
            if job_id in self._memory:
                self._memory[job_id]["state"] = state

"""Single-process local web application with bounded background research jobs."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, field_validator

from .workflow import ResearchError, run

load_dotenv()
logger = logging.getLogger(__name__)
jobs: dict[str, dict] = {}
tasks: dict[str, asyncio.Task] = {}
MAX_ACTIVE = 3
MAX_JOBS = 64


@asynccontextmanager
async def lifespan(app):
    yield
    pending = list(tasks.values())
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()


app = FastAPI(title='Multi-Agent Research Assistant', lifespan=lifespan)


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    mode: Literal['demo', 'extractive', 'ai'] = 'demo'
    limit: int = Field(default=5, ge=1, le=8)

    @field_validator('topic')
    @classmethod
    def nonempty_topic(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError('Enter a topic containing at least 3 characters.')
        return value


async def execute(job_id: str, request: ResearchRequest):
    job = jobs[job_id]
    job['status'] = 'running'

    async def progress(stage):
        job['stage'] = stage

    try:
        async with asyncio.timeout(300):
            result = await run(request.topic, mode=request.mode, limit=request.limit, progress=progress)
        job.update(status='completed', stage='completed', report=result['report'], sources=result['sources'])
    except asyncio.CancelledError:
        job.update(status='cancelled', stage='cancelled', error='Research cancelled.')
    except ResearchError as exc:
        job.update(status='failed', stage='failed', error=str(exc))
    except TimeoutError:
        job.update(status='failed', stage='failed', error='The research timed out after 5 minutes. Try fewer sources or a shorter topic.')
    except Exception:
        logger.exception('Research job failed: %s', job_id)
        job.update(status='failed', stage='failed', error='The research provider failed. Check server credentials and connectivity, or use the offline sample.')
    finally:
        job['finished_at'] = datetime.now(timezone.utc).isoformat()
        tasks.pop(job_id, None)


@app.get('/', response_class=HTMLResponse)
async def home():
    return (Path(__file__).with_name('index.html')).read_text(encoding='utf-8')


@app.get('/health')
async def health():
    return {'status': 'ok', 'active_jobs': len(tasks)}


@app.get('/api/config')
async def config():
    return {'ai_configured': bool(os.getenv('OPENAI_API_KEY')), 'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), 'max_sources': 8}


@app.post('/api/jobs', status_code=202)
async def create_job(request: ResearchRequest):
    if request.mode == 'ai' and not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(503, 'AI mode needs OPENAI_API_KEY. Choose extractive or offline sample mode.')
    if len(tasks) >= MAX_ACTIVE:
        raise HTTPException(429, 'Three research jobs are already active. Wait for one to finish.')
    while len(jobs) >= MAX_JOBS:
        oldest = next((key for key in jobs if key not in tasks), None)
        if oldest is None:
            raise HTTPException(429, 'The job queue is full. Please retry later.')
        jobs.pop(oldest)
    job_id = uuid4().hex
    job = {'id': job_id, 'topic': request.topic, 'mode': request.mode, 'status': 'queued', 'stage': 'queued',
           'created_at': datetime.now(timezone.utc).isoformat(), 'sources': [], 'report': '', 'error': None}
    jobs[job_id] = job
    tasks[job_id] = asyncio.create_task(execute(job_id, request))
    return job


def find_job(job_id):
    if job_id not in jobs:
        raise HTTPException(404, 'Job not found. Jobs expire when the server restarts or its history fills.')
    return jobs[job_id]


@app.get('/api/jobs/{job_id}')
async def get_job(job_id: str):
    return find_job(job_id)


@app.delete('/api/jobs/{job_id}')
async def cancel_job(job_id: str):
    job = find_job(job_id)
    task = tasks.get(job_id)
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # A task cancelled before it starts does not run its finally block.
        tasks.pop(job_id, None)
        job.update(status='cancelled', stage='cancelled', error='Research cancelled.')
    return job


@app.get('/api/jobs/{job_id}/report')
async def download_report(job_id: str):
    job = find_job(job_id)
    if job['status'] != 'completed':
        raise HTTPException(409, 'The report is not ready yet.')
    return Response(job['report'], media_type='text/markdown', headers={'Content-Disposition': f'attachment; filename="research-{job_id}.md"'})

import asyncio
import time
from fastapi.testclient import TestClient
from research_assistant import app as module
from research_assistant.workflow import ResearchError


def wait_for_job(client, job_id):
    for _ in range(100):
        job = client.get('/api/jobs/' + job_id).json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.01)
    raise AssertionError('Job did not finish')


def test_sample_job_completes_and_downloads(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(module.app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/health').json()['status'] == 'ok'
        assert not client.get('/api/config').json()['ai_configured']
        response = client.post('/api/jobs', json={'topic': 'Sample research', 'mode': 'demo', 'limit': 3})
        assert response.status_code == 202
        job = wait_for_job(client, response.json()['id'])
        assert job['status'] == 'completed'
        assert len(job['sources']) == 3
        assert 'OFFLINE SAMPLE' in job['report']
        download = client.get('/api/jobs/' + job['id'] + '/report')
        assert download.status_code == 200
        assert download.text == job['report']
        assert 'attachment' in download.headers['content-disposition']


def test_validation_missing_jobs_and_ai_configuration(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(module.app) as client:
        for payload in ({'topic': '   '}, {'topic': 'valid', 'mode': 'unknown'}, {'topic': 'valid', 'limit': 9}):
            assert client.post('/api/jobs', json=payload).status_code == 422
        assert client.post('/api/jobs', json={'topic': 'valid topic', 'mode': 'ai'}).status_code == 503
        assert client.get('/api/jobs/not-found').status_code == 404


def test_provider_failure_is_reported(monkeypatch):
    async def fail(*args, **kwargs): raise ResearchError('No sources matched that topic.')
    monkeypatch.setattr(module, 'run', fail)
    with TestClient(module.app) as client:
        job_id = client.post('/api/jobs', json={'topic': 'valid topic'}).json()['id']
        job = wait_for_job(client, job_id)
        assert job['status'] == 'failed'
        assert job['error'] == 'No sources matched that topic.'
        assert client.get('/api/jobs/' + job_id + '/report').status_code == 409


def test_cancel_and_capacity(monkeypatch):
    async def slow(*args, **kwargs):
        await asyncio.sleep(100)
    monkeypatch.setattr(module, 'run', slow)
    with TestClient(module.app) as client:
        ids = [client.post('/api/jobs', json={'topic': 'valid topic'}).json()['id'] for _ in range(3)]
        assert client.post('/api/jobs', json={'topic': 'another topic'}).status_code == 429
        assert client.get('/api/jobs/' + ids[0] + '/report').status_code == 409
        assert client.delete('/api/jobs/' + ids[0]).json()['status'] == 'cancelled'
        assert client.get('/health').json()['active_jobs'] == 2
        for job_id in ids[1:]:
            client.delete('/api/jobs/' + job_id)

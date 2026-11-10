import asyncio
import pytest
from research_assistant.workflow import ResearchError, build_workflow, parse_arxiv, run, validate_citations


def source():
    return {'id': 'S1', 'title': 'Paper', 'abstract': 'Evidence', 'url': 'https://arxiv.org/abs/1234.5678', 'published': '2024'}


def test_graph_collects_and_passes_sources_to_each_agent():
    calls = []
    async def search(topic): return [source()]
    async def agent(role, prompt):
        calls.append((role, prompt))
        return role + ' output [S1]'
    result = asyncio.run(build_workflow(agent, search).ainvoke({'topic': 'test'}))
    assert [x[0] for x in calls] == ['researcher', 'critic', 'writer']
    assert all('S1' in x[1] for x in calls)
    assert 'writer output [S1]' in result['report']
    assert source()['url'] in result['report']


def test_no_sources_stops_agents():
    async def search(topic): return []
    with pytest.raises(ResearchError, match='No sources'):
        asyncio.run(build_workflow(search=search).ainvoke({'topic': 'test'}))


def test_rejects_unknown_or_missing_citations():
    for text in ('A claim without a citation.', 'A claim [S999].'):
        with pytest.raises(ResearchError):
            validate_citations(text, [source()])
    validate_citations('A claim [S1].', [source()])


def test_offline_demo_is_explicit_and_requires_no_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    result = asyncio.run(run('test topic', mode='demo', limit=1))
    assert 'OFFLINE SAMPLE' in result['report']
    assert 'fictional' in result['report']
    assert len(result['sources']) == 1
    assert '[S1]' in result['report']


def test_ai_mode_requires_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ResearchError, match='OPENAI_API_KEY'):
        asyncio.run(run('test topic', mode='ai'))


def test_parse_normalizes_sources_and_rejects_error_records():
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.5678v1</id><title> A paper </title><summary>Some evidence.</summary><published>2024-01-01T00:00:00Z</published></entry></feed>'
    records = parse_arxiv(xml)
    assert records[0]['url'] == 'https://arxiv.org/abs/1234.5678v1'
    assert records[0]['published'] == '2024-01-01'
    with pytest.raises(ResearchError):
        parse_arxiv('<feed')
    with pytest.raises(ResearchError):
        parse_arxiv('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#x</id></entry></feed>')


def test_workflow_rejects_fabricated_reference():
    async def search(topic): return [source()]
    async def agent(role, prompt): return 'Made up finding [S99]'
    with pytest.raises(ResearchError, match='not retrieved'):
        asyncio.run(build_workflow(agent, search).ainvoke({'topic': 'test'}))

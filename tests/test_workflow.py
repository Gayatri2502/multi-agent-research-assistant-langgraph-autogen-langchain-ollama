import asyncio
from research_assistant.workflow import build_workflow

def test_graph_collects_and_passes_sources_to_each_agent():
    calls = []
    async def search(topic): return [{'id': 'S1', 'title': 'Paper', 'abstract': 'Evidence', 'url': 'https://arxiv.org/abs/1', 'published': '2024'}]
    async def agent(role, prompt):
        calls.append((role, prompt))
        return role + ' output [S1]'
    result = asyncio.run(build_workflow(agent, search).ainvoke({'topic': 'test'}))
    assert [x[0] for x in calls] == ['researcher', 'critic', 'writer']
    assert all('S1' in x[1] for x in calls)
    assert result['report'] == 'writer output [S1]'

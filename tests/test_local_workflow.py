import asyncio

from langgraph.types import Command
from research_assistant.local_workflow import build_local_graph, check_audit

SOURCE = {'id': 'S1', 'title': 'Evidence study', 'url': 'https://arxiv.org/abs/2309.15217', 'published': '2023',
          'abstract': 'Evaluation requires checking retrieval quality and faithful use of evidence. Abstracts omit detailed experimental methods.'}
DRAFT = '# Research brief\n\n' + ('Evaluation requires checking retrieval quality and faithful use of evidence. ' * 9) + '[S1]\n\n## Limitations\nAbstracts omit detailed experimental methods. [S1]'
AUDIT = {'passed': True, 'issues': [], 'evidence': [{'source_id': 'S1', 'quote': SOURCE['abstract'], 'claim': 'Evaluation requires evidence checks.'}]}


def test_parallel_join_and_real_approval_interrupt():
    async def scenario():
        started = set()
        both = asyncio.Event()
        calls = []
        async def agent(role, prompt):
            calls.append(role)
            if role in ('researcher', 'analyst'):
                started.add(role)
                if len(started) == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), 1)
                return role + ' notes [S1]'
            assert started == {'researcher', 'analyst'}
            assert 'analyst notes' in prompt and 'researcher notes' in prompt
            return DRAFT
        async def audit(prompt):
            assert 'Evaluation requires checking retrieval quality' in prompt and DRAFT in prompt
            return {**AUDIT, 'evidence': [{'passage_id': 'S1:P1', 'claim': 'Evaluation requires evidence checks.'}]}
        async def search(topic): return [SOURCE]
        graph = build_local_graph(agent, audit, search)
        config = {'configurable': {'thread_id': 'test'}}
        result = await graph.ainvoke({'topic': 'evaluation'}, config)
        assert result['__interrupt__']
        assert 'approved' not in result
        final = await graph.ainvoke(Command(resume=True), config)
        assert final['approved'] is True
        assert calls.count('writer') == 1
    asyncio.run(scenario())


def test_failed_qa_stops_after_two_revisions_without_approval():
    async def scenario():
        calls = []
        async def agent(role, prompt):
            calls.append(role)
            return DRAFT if role == 'writer' else 'Notes [S1]'
        async def audit(prompt): return {'passed': False, 'issues': ['Unsupported comparison.'], 'evidence': [{'passage_id': 'S1:P1', 'claim': 'Evaluation'}]}
        async def search(topic): return [SOURCE]
        result = await build_local_graph(agent, audit, search).ainvoke({'topic': 'evaluation'}, {'configurable': {'thread_id': 'retry'}})
        assert calls.count('writer') == 3
        assert not result['audit']['passed']
        assert '__interrupt__' not in result
        assert 'approved' not in result
    asyncio.run(scenario())


def test_quotes_and_citations_are_checked_separately_from_model_verdict():
    forged = {**AUDIT, 'evidence': [{'source_id': 'S1', 'quote': 'A fabricated experimental accuracy of 99 percent.', 'claim': 'Accurate'}]}
    assert not check_audit(forged, DRAFT, [SOURCE])['passed']
    assert not check_audit(AUDIT, DRAFT + ' [S99]', [SOURCE])['passed']
    assert check_audit(AUDIT, DRAFT, [SOURCE])['passed']

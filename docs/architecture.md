# ADR 001 — Local, evidence-led research workflow

Status: Accepted. Date: 2026-11-18.

## Context

The original application collected arXiv abstracts and optionally used OpenAI AutoGen agents sequentially. The goal is a local synthesis workflow with visible evidence, bounded revisions, and real user approval.

## Decision

Use LangGraph for shared state and control flow. After collection, run AutoGen researcher and analyst nodes in parallel, join them before the AutoGen writer, and use LangChain ChatOllama for structured evidence QA. All model calls use the configured local Ollama model. QA receives original sources, never just upstream summaries. Exact quote matching is deterministic; semantic claim support remains a model judgment.

The review router permits an initial draft plus two revisions. Passing QA pauses at a LangGraph interrupt backed by an in-memory checkpointer. Only an explicit user approval resumes to a completed report. Exhausted failures remain unapproved and cannot use final export. Rejection does not publish anything. Approval means local export only, not external publication.

## Alternatives

- OpenAI-only sequential agents: simpler existing code, but requires provider credits and lacks bounded revisions.
- LangGraph plus ChatOllama alone: fewer dependencies and sufficient for fixed tasks, but omits the requested AutoGen agent layer.
- Unrestricted agent chat with web tools: more flexible but harder to bound and inspect, with larger prompt-injection and latency surfaces.

## Consequences

No model API fee for local mode. Topics still leave the machine for arXiv search. Parallel graph scheduling does not guarantee faster inference: Ollama may serialize requests depending on RAM and server configuration. Three frameworks mean more dependencies, so responsibilities stay narrow and explicit. The evidence gate can reject good drafts or accept unsupported interpretations; it is not an independent fact checker.

The service remains loopback-only, single-process, and ephemeral. Checkpoints, jobs, and pending approvals disappear on restart. Persistent storage, authentication, broad web retrieval, full-paper ingestion, and independent evaluation are future work rather than claimed capabilities.

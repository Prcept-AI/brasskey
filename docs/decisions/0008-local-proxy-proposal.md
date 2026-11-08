# 0008: A thin local proxy in front of engines (proposal)

Date: 2026-11-08
Status: accepted (2026-11-08: option 2, built in milestone f; see ADR 0009)

## Context

ADR 0002 decided "no Brasskey proxy in v0.1": engines serve OpenAI-compatible APIs themselves.
Building the MLX adapter (ADR 0007) showed where that falls short:

- `mlx_lm.server` has no API keys, so MLX can never be served beyond loopback;
- a request can make it load a different (cached) model, outside the memory plan;
- `/v1/models` lists every MLX model in the user's cache, not the served model.

Milestone (f) adds API keys for non-localhost binds, and MLX can't meet that rule on its own.
A thin Python proxy for local use was already expected in early versions, ahead of a dedicated
gateway.

## Options

1. **Keep the status quo:** MLX stays loopback-only and the gaps are documented. No new code
   or dependencies. MLX users who need remote access use llama.cpp.
2. **Thin local proxy (recommended), from milestone (f):** Brasskey listens on the requested
   address and forwards to the engine on a private loopback port. In about 300 lines plus tests,
   it adds:
   - API-key checks for every engine, done uniformly;
   - the served model name enforced, with requests for other models rejected;
   - its own `/v1/models` listing;
   - an Origin check, plus request-size and concurrency limits that match the plan;
   - streaming passthrough (server-sent events).

   It needs an ASGI server and framework. Candidates: uvicorn (BSD-3) + Starlette (BSD-3), about
   1 MB, or httpx2's own server support if it has any (to be checked).
3. **Wait for the Rust gateway (v0.5):** no proxy in Python at all; MLX stays loopback-only until
   then.

## Recommendation

Option 2 in milestone (f): it makes the security rules uniform across engines, and it is the
seam the Rust gateway later replaces. It adds dependencies and a component, so it needs your
approval.

## Consequences if accepted

- One extra local hop (sub-millisecond on loopback) and one more process to supervise.
- `doctor --security` can rely on every server having the same auth and Origin behavior.

# fti-agent

MCP Host agent for [Feature Pipeline Hub](../feature_pipeline_hub) — a minimal,
hand-rolled (no LangGraph) model-using-tools-in-a-loop agent that connects to the
hub's MCP server and drives the LoRA dataset curation/training pipeline on behalf
of a human operator.

This is an independent project (own `pyproject.toml`, own `.venv`, own `uv.lock`),
deliberately kept separate from `feature_pipeline_hub/` — that project's own
dependencies are intentionally minimal and framework-agnostic; all the
LangChain/agent-specific tooling lives here instead.

## Setup

```bash
cd agent
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

The agent launches the hub's MCP server itself as a subprocess (`cd
feature_pipeline_hub && uv run python -m mcp_server`, from wherever `agent/` is
checked out relative to `feature_pipeline_hub/` as a sibling directory) — you don't
need to start the server separately, but `feature_pipeline_hub/`'s own `uv sync`
must have been run at least once so its `.venv` exists.

### Using a Remote LLM Server (e.g., LM Studio on macOS, agent on Linux)

If the LLM model runs on a different machine than where the agent runs (e.g., LM
Studio on your Mac, agent running on a Linux VM or server), use an SSH tunnel to
expose the model's API:

**From your Mac, in a dedicated terminal:**

```bash
ssh -N -R 1234:localhost:1234 d13@<linux-ip>
```

Replace `<linux-ip>` with the Linux machine's IP on your LAN (e.g., `192.168.68.52`).
This makes `localhost:1234` on the Linux box point back to LM Studio on the Mac.
Keep this terminal open while you use the agent.

**On the Linux machine, in `agent/.env`:**

```bash
FTI_AGENT_API_BASE=http://localhost:1234/v1
FTI_AGENT_MODEL=openai:qwen3.6-35b-a3b-uncensored-hauhauc-aggressive
OPENAI_API_KEY=lm-studio
```

(Adjust the model name to match what LM Studio is actually serving.)

**Verify both endpoints before running the agent:**

```bash
uv run python validate_connection.py
```

This checks that the model endpoint is reachable AND that the MCP server works,
so you know which leg is broken if something goes wrong.

## Running

```bash
uv run fti-agent                        # interactive REPL
uv run fti-agent --scenario cyberpunk   # fires a canned first message, then REPL
```

Type `exit` or `quit`, or Ctrl-C, to end the session (closes the MCP server
subprocess on the way out).

## Scope

This is phases 1-3 of the agent roadmap only: a real MCP Host connection, a system
prompt tuned to this pipeline's specific tool-sequencing gotchas, and a manual
REPL for the "think like your agent" testing exercise. Predefined workflow
patterns (Orchestrator-Workers, Evaluator-Optimizer) and full autonomous
planning/reflection are future increments, not implemented here.

## Tests

```bash
uv run pytest   # no LLM API key required — tests only tool discovery/round-trip
uv run mypy
```

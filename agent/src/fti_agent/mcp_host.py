"""MCP Host: launches the hub's MCP server as a subprocess and discovers its tools.

Reproduces `cd feature_pipeline_hub && uv run python -m mcp_server` (the command
documented in that project's own CLAUDE.md/README) over stdio, from wherever this
project happens to be checked out relative to it.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fti_agent.config import AgentConfig


def hub_server_params(
    config: AgentConfig, *, env_overrides: dict[str, str] | None = None
) -> StdioServerParameters:
    """`env_overrides` is for tests that need to point the hub at a scratch DB
    (`FTI_DB_PATH`) without touching the real one — this is a fully trusted
    local subprocess, so the full parent environment is passed through rather
    than an allowlisted subset.
    """
    env = {**os.environ, **(env_overrides or {})}
    return StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "mcp_server"],
        cwd=str(config.hub_dir),
        env=env,
    )


@asynccontextmanager
async def mcp_session(
    config: AgentConfig, *, env_overrides: dict[str, str] | None = None
) -> AsyncIterator[tuple[ClientSession, list[BaseTool]]]:
    """Open one MCP session for the whole agent run and discover its tools.

    Kept open for the lifetime of the caller's `async with` block rather than
    reopened per turn — the subprocess (and its own per-operation SQLite
    connections, owned server-side) stays alive across a multi-turn REPL session.
    """
    async with stdio_client(hub_server_params(config, env_overrides=env_overrides)) as (
        read,
        write,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            yield session, tools

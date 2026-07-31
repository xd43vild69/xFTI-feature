"""Smoke tests for MCP tool discovery — no LLM API key required.

Launches the real hub MCP server as a real stdio subprocess (the same path
production uses), pointed at a scratch SQLite DB via FTI_DB_PATH so it never
touches the developer's real data.
"""

from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from fti_agent.config import load_config
from fti_agent.mcp_host import mcp_session

EXPECTED_TOOL_NAMES = {
    "list_dataset_runs",
    "get_dataset_health",
    "get_run_detail",
    "import_dataset",
    "revalidate_run",
    "export_dataset",
    "quality_summary",
    "start_lora_training",
    "continue_lora_training",
    "get_training_status",
    "stop_training",
}


@pytest.fixture
def scratch_db_env(tmp_path: Path) -> dict[str, str]:
    return {"FTI_DB_PATH": str(tmp_path / "test.db")}


async def test_discovers_all_eleven_tools_with_descriptions(scratch_db_env: dict[str, str]) -> None:
    config = load_config()

    async with mcp_session(config, env_overrides=scratch_db_env) as (_session, tools):
        names = {tool.name for tool in tools}
        assert names == EXPECTED_TOOL_NAMES
        for tool in tools:
            assert tool.description, f"{tool.name} lost its docstring in MCP->LangChain conversion"


async def test_a_discovered_tool_round_trips_a_real_call(scratch_db_env: dict[str, str]) -> None:
    config = load_config()

    async with mcp_session(config, env_overrides=scratch_db_env) as (_session, tools):
        list_runs = next(t for t in tools if t.name == "list_dataset_runs")
        call = {"name": "list_dataset_runs", "args": {}, "id": "call_1", "type": "tool_call"}

        result = await list_runs.ainvoke(call)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call_1"
        assert getattr(result, "status", "success") != "error"
        assert result.content == []


async def test_a_tool_error_comes_back_as_an_error_tool_message(scratch_db_env: dict[str, str]) -> None:
    config = load_config()

    async with mcp_session(config, env_overrides=scratch_db_env) as (_session, tools):
        get_run_detail = next(t for t in tools if t.name == "get_run_detail")
        call = {
            "name": "get_run_detail",
            "args": {"run_id": "does-not-exist"},
            "id": "call_2",
            "type": "tool_call",
        }

        result = await get_run_detail.ainvoke(call)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "does-not-exist" in str(result.content)

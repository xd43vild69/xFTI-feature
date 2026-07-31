"""Preflight check: is the model endpoint reachable, and does the hub's MCP server work?

Checks both legs independently so a failure points at one of them, not "something
broke". Costs one trivial model call, so it also catches a bad key or a model name
the endpoint doesn't actually serve.
"""

import asyncio
import sys

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from fti_agent.config import AgentConfig, load_config
from fti_agent.mcp_host import mcp_session


def build_model(config: AgentConfig) -> BaseChatModel:
    if config.api_base:
        model_name = config.model.split(":")[-1] if ":" in config.model else config.model
        return ChatOpenAI(
            base_url=config.api_base,
            api_key="lm-studio",  # type: ignore[arg-type]
            model=model_name,
        )
    return init_chat_model(config.model)


async def check_model(config: AgentConfig) -> bool:
    endpoint = config.api_base or "provider default"
    print(f"[1/2] Model {config.model} at {endpoint}")
    try:
        reply = await build_model(config).ainvoke([HumanMessage("Reply with just: ok")])
        print(f"      OK — replied {str(reply.content)[:60]!r}")
        return True
    except Exception as exc:
        print(f"      FAILED — {type(exc).__name__}: {exc}")
        if config.api_base and "localhost" in config.api_base:
            print("      Note: 'localhost' is this machine. If the model server runs")
            print("      elsewhere (another host on the LAN), use its IP instead.")
        return False


async def check_mcp(config: AgentConfig) -> bool:
    print(f"[2/2] MCP server at {config.hub_dir}")
    try:
        async with mcp_session(config) as (_session, tools):
            print(f"      OK — {len(tools)} tools: {', '.join(sorted(t.name for t in tools))}")
            return True
    except Exception as exc:
        print(f"      FAILED — {type(exc).__name__}: {exc}")
        return False


async def main() -> None:
    config = load_config()
    model_ok = await check_model(config)
    mcp_ok = await check_mcp(config)

    print()
    if model_ok and mcp_ok:
        print("Both legs up — `uv run fti-agent` should work.")
    else:
        print("Not ready: fix the FAILED leg above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

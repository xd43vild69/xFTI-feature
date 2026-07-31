"""Interactive REPL: the "think like your agent" testing surface.

Opens one MCP session for the whole process, prints every tool call and tool
result as they happen (not just the final answer), and keeps conversation state
across REPL turns so a full start -> poll -> continue training flow can be
exercised in one sitting.
"""

import argparse
import asyncio
import json
import sys

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from fti_agent.config import load_config
from fti_agent.loop import AgentLoop, TurnResult
from fti_agent.mcp_host import mcp_session
from fti_agent.system_prompt import SYSTEM_PROMPT

SCENARIOS = {
    "cyberpunk": "Prepara un dataset para el estilo Cyberpunk y lanza el entrenamiento.",
}


def _print_tool_call(call: dict) -> None:
    args = json.dumps(call.get("args", {}), ensure_ascii=False)
    print(f"\n[tool call] {call['name']}({args})")


def _print_tool_result(message: ToolMessage) -> None:
    label = "tool error" if getattr(message, "status", None) == "error" else "tool result"
    content = message.content
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    if len(content) > 2000:
        content = content[:2000] + "… (truncated)"
    print(f"[{label}] {content}")


def _print_turn_result(result: TurnResult) -> None:
    if result.hit_iteration_cap:
        print(
            f"\n[loop] Hit the iteration cap without a final answer — "
            f"stopping to avoid a runaway loop."
        )
        return
    print(f"\n[assistant] {result.final_text}")


async def _run(scenario: str | None) -> None:
    config = load_config()
    if config.api_base:
        model_name = config.model.split(":")[-1] if ":" in config.model else config.model
        model: BaseChatModel = ChatOpenAI(
            base_url=config.api_base,
            api_key="lm-studio",  # type: ignore[arg-type]
            model=model_name,
        )
    else:
        model = init_chat_model(config.model)

    async with mcp_session(config) as (_session, tools):
        loop = AgentLoop(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            max_iterations=config.max_iterations,
        )

        def on_event(kind: str, payload: object) -> None:
            if kind == "tool_call":
                _print_tool_call(payload)  # type: ignore[arg-type]
            elif kind == "tool_result":
                _print_tool_result(payload)  # type: ignore[arg-type]

        pending_first_message = SCENARIOS.get(scenario) if scenario else None

        print(f"fti-agent ready ({len(tools)} tools, model={config.model}). "
              "Type 'exit' or 'quit' to stop.\n")

        while True:
            if pending_first_message is not None:
                user_text = pending_first_message
                pending_first_message = None
                print(f"> {user_text}")
            else:
                try:
                    user_text = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    break

            try:
                result = await loop.run_turn(user_text, on_event=on_event)
            except Exception as exc:
                # A model-call failure (unreachable endpoint, bad key, rate limit)
                # is a bad turn, not a bad session — keep the MCP subprocess and
                # the conversation alive so the user can fix it and retry.
                print(f"\n[error] {type(exc).__name__}: {exc}")
                if config.api_base:
                    print(f"[error] Model endpoint is {config.api_base} — is it reachable from here?")
                continue
            _print_turn_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="fti-agent: MCP Host REPL for Feature Pipeline Hub")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default=None,
        help="Fire a canned first message, then drop into the REPL.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args.scenario))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

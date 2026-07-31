"""The agent loop itself: a model repeatedly using tools, driven by hand.

No LangGraph, no state machine — just "call the model, run whatever tools it
asked for, feed the results back, repeat until it stops asking" per Anthropic's
own guidance to start as simple as possible. `load_mcp_tools` (mcp_host.py)
already converts MCP-level tool errors into `ToolMessage(status="error")`
instead of raising, so the model sees failures as ordinary turns it can react
to — this loop's own try/except is only a fallback for anything that still
raises (a dropped connection, a malformed tool-call).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

EventSink = Callable[[str, object], None]


@dataclass
class TurnResult:
    messages: list[BaseMessage]
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolMessage] = field(default_factory=list)
    final_text: str | None = None
    hit_iteration_cap: bool = False


class AgentLoop:
    """Owns the running conversation and drives one user turn at a time."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str,
        max_iterations: int,
    ) -> None:
        self._model = model.bind_tools(tools)
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._max_iterations = max_iterations
        self._history: list[BaseMessage] = [SystemMessage(system_prompt)]

    async def run_turn(self, user_text: str, on_event: EventSink | None = None) -> TurnResult:
        """Run one user turn to completion: model calls, tool calls, repeat.

        `on_event` (if given) is called synchronously for each tool call and
        tool result as they happen, for a caller (the CLI) that wants to print
        progress rather than wait for the whole turn to finish.
        """
        self._history.append(HumanMessage(user_text))
        result = TurnResult(messages=[])

        for _ in range(self._max_iterations):
            ai_message = await self._model.ainvoke(self._history)
            self._history.append(ai_message)
            result.messages.append(ai_message)

            if not ai_message.tool_calls:
                result.final_text = _text_of(ai_message)
                return result

            for call in ai_message.tool_calls:
                result.tool_calls.append(call)
                if on_event is not None:
                    on_event("tool_call", call)

                tool_message = await self._run_one_tool(call)
                result.tool_results.append(tool_message)
                self._history.append(tool_message)
                if on_event is not None:
                    on_event("tool_result", tool_message)

        result.hit_iteration_cap = True
        return result

    async def _run_one_tool(self, call: ToolCall) -> ToolMessage:
        tool = self._tools_by_name.get(call["name"])
        if tool is None:
            return ToolMessage(
                content=f"Unknown tool: {call['name']}",
                tool_call_id=call["id"],
                status="error",
            )
        try:
            output = await tool.ainvoke(call)
        except Exception as exc:  # transport/session failures load_mcp_tools doesn't catch
            return ToolMessage(content=str(exc), tool_call_id=call["id"], status="error")
        if isinstance(output, ToolMessage):
            return output
        return ToolMessage(content=str(output), tool_call_id=call["id"])


def _text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "") for block in message.content if isinstance(block, dict)
    )

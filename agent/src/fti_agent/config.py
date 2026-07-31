"""Agent configuration: model selection, the hub's location, and loop limits."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "anthropic:claude-sonnet-5"
DEFAULT_MAX_ITERATIONS = 15


@dataclass(frozen=True)
class AgentConfig:
    model: str
    hub_dir: Path
    max_iterations: int
    api_base: str | None = None


def load_config() -> AgentConfig:
    """Read config from `agent/.env` (if present) and the environment.

    `hub_dir` defaults to the `feature_pipeline_hub/` sibling directory next to
    this project — resolved from this file's own location, not the process's
    cwd, since the CLI may be launched from anywhere.
    """
    load_dotenv()

    hub_dir_override = os.environ.get("FTI_AGENT_HUB_DIR", "").strip()
    hub_dir = (
        Path(hub_dir_override)
        if hub_dir_override
        else Path(__file__).resolve().parents[3] / "feature_pipeline_hub"
    )
    if not hub_dir.is_dir():
        raise FileNotFoundError(
            f"feature_pipeline_hub not found at {hub_dir} — set FTI_AGENT_HUB_DIR "
            "if it's not a sibling of this agent/ project."
        )

    return AgentConfig(
        model=os.environ.get("FTI_AGENT_MODEL", DEFAULT_MODEL),
        hub_dir=hub_dir,
        max_iterations=int(os.environ.get("FTI_AGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)),
        api_base=os.environ.get("FTI_AGENT_API_BASE"),
    )

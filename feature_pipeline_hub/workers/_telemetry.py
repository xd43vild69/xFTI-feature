"""JSON Lines lifecycle events for the vendored, detached workers.

precache_worker.py and train_worker.py are ported byte-for-byte from LoRAlab
(see their own docstrings) so they stay easy to diff against the upstream
project as it evolves — nothing in their bodies should be touched to add
logging. This module is imported only from their `if __name__ == "__main__":`
blocks, wrapping the vendored entrypoint call rather than instrumenting it:
the free-text log output stays byte-identical, and one structured event per
lifecycle transition rides alongside it on the same stdout stream.

Mirrors the WorkerLifecycleEvent schema in
src/feature_pipeline/domain/worker_contracts.py — kept as plain dicts here
because these workers run in training_runtime/venv, a separate interpreter
from the hub's own (pydantic-free by design, same reason recaption_worker.py
builds its events as dicts instead of importing the hub's models).
"""

import json
import os
import time


def emit_lifecycle(event: str, worker: str, **fields) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "worker": worker,
                "run_id": os.environ.get("FTI_RUN_ID", ""),
                "timestamp": time.time(),
                **fields,
            }
        ),
        flush=True,
    )

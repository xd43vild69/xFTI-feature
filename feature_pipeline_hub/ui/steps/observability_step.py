"""Step 0: Pipeline observability dashboard.

Shows aggregated metrics across all 5 curation + training steps: durations,
throughput, errors, and GPU costs. Helps operators understand the pipeline
performance and diagnose bottlenecks.

This step can be viewed anytime and is updated as each step completes.
"""

from components import observability_panel

observability_panel.render()

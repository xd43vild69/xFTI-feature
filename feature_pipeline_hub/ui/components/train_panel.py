"""Step 5: launch and monitor a training run against the exported dataset.

Pre-cache is short (I/O + VAE-encode bound) and runs synchronously, blocking
the Start button while it finishes; training itself is hours long and runs
detached, so progress is read back from disk/SQLite on every refresh — this
page works the same whether it's the tab that launched the run or a fresh
browser session hours later.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

import state
from feature_pipeline.application import dataset_service, export_service, training_service
from feature_pipeline.domain import naming
from feature_pipeline.domain.curation_report import Tier, WeightProfile, tier_counts
from feature_pipeline.domain.models import DatasetSample, IngestionRun
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.database import get_connection

LOG_TAIL_BYTES = 8000


@contextmanager
def _db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _launch_config_state(run_id: str) -> dict:
    key = f"train_launch_config_{run_id}"
    if key not in st.session_state:
        st.session_state[key] = training_service.TrainingConfig().model_dump()
    return st.session_state[key]


def _field_key(run_id: str, name: str) -> str:
    version = st.session_state.get(f"train_launch_field_version_{run_id}", 0)
    return f"train_field_{name}_v{version}"


def _json_key(run_id: str) -> str:
    version = st.session_state.get(f"train_launch_json_version_{run_id}", 0)
    return f"train_json_v{version}"


def _bump(key: str) -> None:
    st.session_state[key] = st.session_state.get(key, 0) + 1


def _sync_fields_to_json(run_id: str) -> None:
    """on_change callback for every field widget: fold its new value into the
    canonical config and force the JSON tab's textarea to remount with it."""
    config = _launch_config_state(run_id)
    for name in training_service.TrainingConfig.model_fields:
        widget_key = _field_key(run_id, name)
        if widget_key in st.session_state:
            config[name] = st.session_state[widget_key]
    _bump(f"train_launch_json_version_{run_id}")
    st.session_state[f"train_launch_json_error_{run_id}"] = None


def _sync_json_to_fields(run_id: str) -> None:
    """on_change callback for the JSON textarea: parse, validate, merge, and force
    the field widgets to remount with the result — or leave everything untouched
    and surface an error if the pasted text doesn't parse/validate."""
    raw = st.session_state[_json_key(run_id)]
    error_key = f"train_launch_json_error_{run_id}"
    try:
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise ValueError("Blueprint must be a JSON object of field: value pairs.")
        config = _launch_config_state(run_id)
        new_config, extra_keys = training_service.merge_training_config_overrides(
            config, overrides
        )
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        st.session_state[error_key] = f"Invalid JSON blueprint: {exc}"
        return
    config.update(new_config)
    st.session_state[error_key] = (
        f"Ignored unknown field(s): {', '.join(extra_keys)}" if extra_keys else None
    )
    _bump(f"train_launch_field_version_{run_id}")


MODE_NEW = "New training"
MODE_RESUME = "Resume"
MODE_BRANCH = "Branch"
MODE_MONITOR = "Monitor"
_MODES = (MODE_NEW, MODE_RESUME, MODE_BRANCH, MODE_MONITOR)
_MODE_ICONS = {
    MODE_NEW: ":material/play_arrow:",
    MODE_RESUME: ":material/fast_forward:",
    MODE_BRANCH: ":material/call_split:",
    MODE_MONITOR: ":material/monitoring:",
}


_BUSY_HELP = (
    "The GPU is already running a job. Wait for it to finish (see Monitor) — you can "
    "keep configuring this launch in the meantime."
)


def _mode_key(run_id: str) -> str:
    # Per dataset: a mode chosen on one dataset should not carry over to another where
    # it may not apply (no checkpoints, no runs to monitor).
    return f"train_mode_{run_id}"


def _go_to_monitor(run_id: str) -> None:
    """Hand off to the Monitor after a launch, then rerun.

    Explicit because the mode selector is keyed: without this the operator would be left
    looking at the form they just submitted. The old page switched by accident, since
    render() checked the newest run's status before drawing anything.
    """
    st.session_state[_mode_key(run_id)] = MODE_MONITOR
    st.rerun()


def render() -> None:
    """Four independent flows, one visible at a time.

    They are alternatives, not steps, and rendering them stacked (as this page did)
    reads as a sequence. Only the selected one's body executes, which also keeps the
    branch form's per-image data_editor from being built while someone is editing
    hyperparameters.
    """
    run = state.require_active_run()
    if run is None:
        return

    with _db() as conn:
        train_runs = [
            r for r in training_repo.list_training_runs(conn, dataset_run_id=run.run_id)
            if r.kind == "train"
        ]
        fork_points = training_service.find_fork_points(conn, dataset_run_id=run.run_id)
    # Resume only ever applies to a run's own restorable state; the weights-only points
    # find_fork_points adds are forkable but not resumable, so they are filtered out
    # here rather than offered and then refused.
    resume_points = [p for p in fork_points if p.kind == "exact"]
    active = state.active_training_run()

    _render_status_strip(train_runs, active)
    mode = _render_mode_selector(run, active)

    # The Monitor is resolved before the launch preconditions: an unavailable runtime, a
    # missing export or a stale trigger word are all reasons not to *launch*, none of
    # them a reason to be unable to read the log of what already ran.
    if mode == MODE_MONITOR:
        _render_monitor(run, train_runs)
        return

    if _render_launch_blocker(run):
        return

    busy = active is not None
    if mode == MODE_RESUME:
        _render_resume_form(run, resume_points, busy=busy)
    elif mode == MODE_BRANCH:
        _render_fork_form(run, fork_points, busy=busy)
    else:
        _render_new_run_form(run, busy=busy)


def _render_status_strip(
    train_runs: list[training_repo.TrainingRun],
    active: training_repo.TrainingRun | None,
) -> None:
    """The one line that stays visible in every mode: what the GPU is doing."""
    if active is not None:
        where = "this dataset" if any(
            r.training_run_id == active.training_run_id for r in train_runs
        ) else "another dataset"
        st.warning(
            f":material/bolt: GPU busy — a **{active.kind}** job for {where} has been "
            f"running since {active.started_at:%Y-%m-%d %H:%M}. Launching is disabled "
            f"until it finishes.",
        )
        return
    if train_runs:
        latest = train_runs[0]  # newest first
        st.caption(f"Last run: {latest.status} · {latest.finished_at or 'in progress'}")


def _render_mode_selector(
    run: IngestionRun, active: training_repo.TrainingRun | None
) -> str:
    """The four modes, always all four.

    Offering them unconditionally costs an explanation inside the empty ones, and buys a
    selector whose shape does not change under the operator — and a way to find out that
    branching exists before there is a checkpoint to branch from.
    """
    key = _mode_key(run.run_id)
    if key not in st.session_state:
        # Only ever the *initial* mode; once the operator picks one it is theirs to keep.
        st.session_state[key] = MODE_MONITOR if active is not None else MODE_NEW

    mode = st.segmented_control(
        "Mode",
        _MODES,
        format_func=lambda name: f"{_MODE_ICONS[name]} {name}",
        key=key,
        # Without this the active option can be clicked off, leaving no mode selected
        # and the page showing a body nothing in the bar points at.
        required=True,
        label_visibility="collapsed",
    )
    return str(mode) if mode else MODE_NEW


def _render_launch_blocker(run: IngestionRun) -> bool:
    """Render whatever stops this dataset from being trained, if anything.

    Returns True when it rendered something and the caller must not draw a launch form.
    """
    try:
        training_runner.resolve_environment()
    except training_runner.TrainingUnavailable as exc:
        st.caption(f":material/info: {exc}")
        return True

    dataset_dir = training_service.dataset_dir_for(run.concept.concept_name)
    if not dataset_dir.is_dir() or not any(dataset_dir.iterdir()):
        st.caption(
            f":material/info: No exported dataset at training_runtime/datasets/"
            f"{run.concept.concept_name}/. Export it in Step 4 first."
        )
        return True

    conflict = training_service.detect_dataset_version_conflict(run.concept.concept_name)
    dismissed_key = f"version_conflict_dismissed_{run.run_id}"
    if conflict is not None and not st.session_state.get(dismissed_key):
        _render_version_conflict_notice(conflict, dismissed_key)
        return True

    return False


def _render_version_conflict_notice(
    conflict: training_service.DatasetVersionConflict, dismissed_key: str
) -> None:
    """Block launching until the operator resolves a stale versioned trigger word.

    `{prefix}_v1` text surviving into `{prefix}_v2`'s captions almost always means
    the images/captions were copied forward from the older export rather than
    recaptured — training on them as-is would bake the wrong trigger word into the
    adapter, so this stops short of the hyperparameter form until it's addressed.
    """
    st.warning(
        f":material/warning: {conflict.affected_files} caption(s) in "
        f"**{conflict.current_name}** still say **{conflict.stale_trigger_word}** — "
        f"looks like this dataset was copied forward from an earlier version."
    )

    with st.container(horizontal=True):
        overwrite = st.button(
            f"Update captions to {conflict.current_name}", icon=":material/edit:"
        )
        increase = st.button(
            f"Update + suggest next version ({conflict.suggested_next_version})",
            icon=":material/add:",
        )
        ignore = st.button("Ignore, use captions as-is", icon=":material/close:")

    if overwrite or increase:
        dataset_dir = training_service.dataset_dir_for(conflict.current_name)
        updated = training_service.update_captions_in_dataset_dir(
            dataset_dir, conflict.stale_trigger_word, conflict.current_name
        )
        st.success(f"Updated {updated} caption(s) to {conflict.current_name}.")
        if increase:
            st.info(
                f"For your next import, name the dataset "
                f"**{conflict.suggested_next_version}** to keep following the sequence."
            )
        st.rerun()

    if ignore:
        st.session_state[dismissed_key] = True
        st.rerun()


def _render_new_run_form(run: IngestionRun, *, busy: bool) -> None:
    """Train this dataset from step 0, with every hyperparameter open to change.

    The only mode that collects the full set — resume and branch both pin them to the
    checkpoint they continue from, since a changed rank or alpha makes the saved adapter
    fail to load.
    """
    st.caption(
        "Starts from scratch on the exported dataset. Everything below is yours to set; "
        "the checkpoints it writes become the starting points offered by Resume and "
        "Branch."
    )

    config = _launch_config_state(run.run_id)

    st.text_input(
        "Checkpoint name",
        value=config["checkpoint_name"] or run.concept.concept_name,
        key=_field_key(run.run_id, "checkpoint_name"),
        on_change=_sync_fields_to_json, args=(run.run_id,),
        help=(
            "Nombre base de los archivos .safetensors generados: "
            "{nombre}_step_N.safetensors y {nombre}_FINAL.safetensors. "
            "Obligatorio — se sanea automáticamente a minúsculas y guiones bajos."
        ),
    )

    form_tab, json_tab = st.tabs(["Form", "JSON"])

    with form_tab:
        with st.container(horizontal=True):
            st.number_input(
                "Total steps", min_value=1, step=100,
                value=config["total_steps"], key=_field_key(run.run_id, "total_steps"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Número total de micro-pasos de entrenamiento. Las actualizaciones "
                    "reales del modelo son total_steps / grad_accum_steps. Más pasos = "
                    "más tiempo de aprendizaje pero más riesgo de sobreajuste en "
                    "datasets pequeños."
                ),
            )
            st.number_input(
                "Learning rate", min_value=0.0, step=1e-5, format="%.6f",
                value=config["lr"], key=_field_key(run.run_id, "lr"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Tasa de aprendizaje máxima, alcanzada tras el warmup y luego "
                    "decaída según el scheduler. Más alta aprende más rápido pero "
                    "puede volverse inestable; más baja es más lenta pero más estable."
                ),
            )
            st.number_input(
                "LoRA rank", min_value=1, step=1,
                value=config["lora_rank"], key=_field_key(run.run_id, "lora_rank"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Capacidad del adaptador LoRA. Rank bajo (4-8) es rápido, liviano "
                    "y menos propenso a sobreajuste en datasets chicos; rank alto "
                    "(32-64+) captura transformaciones más complejas pero usa más "
                    "VRAM y sobreajusta más fácil."
                ),
            )
            st.number_input(
                "LoRA alpha", min_value=1, step=1,
                value=config["lora_alpha"], key=_field_key(run.run_id, "lora_alpha"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Junto con el rank define la escala (alpha/rank) con la que el "
                    "adaptador se aplica al modelo base. Subir alpha sin subir el "
                    "rank intensifica cuánto 'jala' el adaptador — útil para que el "
                    "estilo domine, pero si se exagera degrada la coherencia general. "
                    "Convención habitual: alpha ≈ 2×rank."
                ),
            )

        with st.container(horizontal=True):
            st.number_input(
                "Batch size", min_value=1, step=1,
                value=config["batch_size"], key=_field_key(run.run_id, "batch_size"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Imágenes procesadas juntas por micro-paso. Más alto da un "
                    "gradiente más estable pero usa más VRAM; en GPUs limitadas se "
                    "deja en 1 y se compensa con grad_accum_steps."
                ),
            )
            st.number_input(
                "Grad accumulation steps", min_value=1, step=1,
                value=config["grad_accum_steps"], key=_field_key(run.run_id, "grad_accum_steps"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Micro-pasos acumulados antes de cada actualización real del "
                    "optimizador. Subirlo simula un batch más grande sin más VRAM, "
                    "pero cada actualización tarda más en llegar."
                ),
            )
            st.number_input(
                "Save every", min_value=1, step=25,
                value=config["save_every"], key=_field_key(run.run_id, "save_every"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Cada cuántos micro-pasos se guarda un checkpoint completo "
                    "(modelo, optimizador, EMA, RNG). Más bajo da más seguridad ante "
                    "fallos pero más uso de disco/I-O; más alto reduce overhead pero "
                    "arriesga más trabajo perdido si se interrumpe."
                ),
            )
            st.number_input(
                "Seed", min_value=0, step=1,
                value=config["seed"], key=_field_key(run.run_id, "seed"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Semilla aleatoria que fija torch/CUDA/numpy y el muestreo de "
                    "ruido. Con el mismo seed y los mismos pasos, dos corridas dan "
                    "resultados idénticos — útil para comparar cambios de forma "
                    "controlada."
                ),
            )

        with st.container(horizontal=True):
            st.number_input(
                "Warmup steps", min_value=0, step=10,
                value=config["warmup_steps"], key=_field_key(run.run_id, "warmup_steps"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Actualizaciones iniciales donde la tasa de aprendizaje sube "
                    "gradualmente desde 0 hasta lr. Si el valor iguala o supera el "
                    "total de actualizaciones, el sistema lo recorta automáticamente "
                    "al 10% del total."
                ),
            )
            st.selectbox(
                "LR scheduler", ["cosine", "constant", "linear", "cosine_with_restarts", "step"],
                key=_field_key(run.run_id, "lr_scheduler"),
                index=["cosine", "constant", "linear", "cosine_with_restarts", "step"].index(
                    config["lr_scheduler"]
                ),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Cómo decae la tasa de aprendizaje tras el warmup: cosine (curva "
                    "suave, la más usada), constant (fija en lr), linear (línea "
                    "recta), cosine_with_restarts (repite la curva varias veces, ver "
                    "'LR restarts') o step (baja en saltos)."
                ),
            )
            st.number_input(
                "LR restarts (cosine_with_restarts only)", min_value=1, step=1,
                value=config["lr_num_cycles"], key=_field_key(run.run_id, "lr_num_cycles"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Solo aplica con cosine_with_restarts: cuántas veces se repite "
                    "el ciclo de subida/bajada de la tasa de aprendizaje. Más ciclos "
                    "dan más 'reinicios' (más exploración, afinado final menos "
                    "suave); sin efecto con otros schedulers."
                ),
            )

        with st.container(horizontal=True):
            st.selectbox(
                "Timestep weighting", ["none", "bell", "half_bell"],
                key=_field_key(run.run_id, "timestep_weighting"),
                index=["none", "bell", "half_bell"].index(config["timestep_weighting"]),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Cómo se pondera la pérdida según el nivel de ruido de cada "
                    "muestra: none pondera igual todos los niveles, bell da más peso "
                    "a niveles intermedios, half_bell hace lo mismo solo en la mitad "
                    "inferior. Los pesos están normalizados para no alterar la tasa "
                    "de aprendizaje efectiva."
                ),
            )
            st.number_input(
                "Noise offset", min_value=0.0, step=0.01, format="%.3f",
                value=config["noise_offset"], key=_field_key(run.run_id, "noise_offset"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Agrega un desplazamiento constante al ruido de entrenamiento. "
                    "Desaconsejado para este modelo (usa rectified flow, donde el "
                    "ruido puro ya es la distribución esperada) — se deja en 0 salvo "
                    "que se replique una configuración externa que lo use."
                ),
            )
            st.number_input(
                "Caption dropout rate", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                value=config["caption_dropout_rate"],
                key=_field_key(run.run_id, "caption_dropout_rate"),
                on_change=_sync_fields_to_json, args=(run.run_id,),
                help=(
                    "Probabilidad de que, en un micro-paso, se use el prompt vacío "
                    "en vez del caption real. Subirlo mejora la fidelidad al prompt "
                    "en inferencia (como classifier-free guidance) pero reduce la "
                    "señal de aprendizaje del caption en cada paso."
                ),
            )

        if config["noise_offset"] > 0:
            st.caption(
                ":material/warning: noise_offset is discouraged under rectified flow "
                "(Krea2's own math_ops.py docstring) — usually leave at 0 for this model."
            )

    with json_tab:
        st.text_area(
            "Hyperparameter blueprint (JSON)",
            value=json.dumps(config, indent=2),
            key=_json_key(run.run_id),
            height=320,
            on_change=_sync_json_to_fields, args=(run.run_id,),
        )
        error = st.session_state.get(f"train_launch_json_error_{run.run_id}")
        if error:
            if error.startswith("Ignored unknown"):
                st.warning(error)
            else:
                st.error(error)

    if st.button(
        "Start training", icon=":material/play_arrow:",
        disabled=busy, help=_BUSY_HELP if busy else None,
    ):
        checkpoint_name = naming.slugify(config["checkpoint_name"] or "")
        if not checkpoint_name:
            st.error("El nombre del checkpoint es obligatorio.")
            return
        config["checkpoint_name"] = checkpoint_name
        try:
            validated_config = training_service.TrainingConfig(**config)
        except ValidationError as exc:
            st.error(str(exc))
            return
        with st.spinner("Pre-caching dataset…"):
            with _db() as conn:
                try:
                    training_service.start_training(
                        conn,
                        dataset_run_id=run.run_id,
                        dataset_name=run.concept.concept_name,
                        trigger_word=run.concept.trigger_word,
                        config=validated_config,
                    )
                except training_service.PrecacheFailed as exc:
                    st.error(str(exc))
                    return
        _go_to_monitor(run.run_id)


def _render_resume_form(
    run: IngestionRun, resume_points: list[training_service.ResumePoint], *, busy: bool
) -> None:
    """Offer to continue a previous run instead of starting over from step 0.

    Only `total_steps` is editable: the checkpoint's adapter was shaped by the
    original rank/alpha and would fail to load against different ones, so the
    rest of the hyperparameters are shown read-only rather than re-collected.
    """
    st.caption(
        "Picks a previous run back up where it stopped, in its own output directory — "
        "the optimizer, EMA, RNG and sampler position all restored, and its logs "
        "continued rather than replaced. To leave the original intact and vary the "
        "data instead, use Branch."
    )

    if not resume_points:
        st.info(
            "No checkpoint to resume from yet. Launch a run from **New training** — the "
            "first one lands after `save_every` micro-steps, and this list fills in from "
            "there."
        )
        return

    point = st.selectbox(
        "Checkpoint",
        resume_points,
        format_func=lambda p: (
            f"{_fork_lineage_label(p)} · step {p.step} of {p.total_steps} · "
            f"{p.status} · {p.started_at:%Y-%m-%d %H:%M}"
        ),
    )
    if point is None:
        return

    st.caption(
        f"Resumes at step {point.step + 1} with the optimizer, EMA, RNG and sampler "
        f"position restored · rank {point.config.get('lora_rank')} · "
        f"alpha {point.config.get('lora_alpha')} · lr {point.config.get('lr')} · "
        f"batch {point.config.get('batch_size')} · {point.output_dir}"
    )

    with st.container(horizontal=True):
        total_steps = st.number_input(
            "Total steps",
            min_value=point.step + 1,
            value=max(point.total_steps, point.step + 1),
            step=100,
            key="resume_total_steps",
            help="Must exceed the checkpoint's step, or there is nothing left to train.",
        )
        resume_clicked = st.button(
            f"Resume from step {point.step}", icon=":material/fast_forward:",
            disabled=busy, help=_BUSY_HELP if busy else None,
        )

    if resume_clicked:
        with st.spinner("Pre-caching dataset…"):
            with _db() as conn:
                try:
                    training_service.resume_training(
                        conn,
                        dataset_run_id=run.run_id,
                        resume_point=point,
                        total_steps=int(total_steps),
                    )
                except (
                    training_service.PrecacheFailed,
                    training_service.ResumeUnavailable,
                ) as exc:
                    st.error(str(exc))
                    return
        _go_to_monitor(run.run_id)


_FORK_PINNED_LABELS = (
    "lr", "lora_rank", "lora_alpha", "seed", "batch_size", "grad_accum_steps",
    "lr_scheduler", "warmup_steps",
)


def _fork_images_dataframe(samples: list[DatasetSample]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in samples],
            "incluir": [not s.is_excluded for s in samples],
            "imagen": [Path(s.image_path).stem for s in samples],
            "tier": ["good" for _ in samples],
            "caption": [s.caption for s in samples],
        }
    )


def _fork_lineage_label(point: training_service.ResumePoint) -> str:
    """Name one training well enough to tell it from its siblings.

    Two parts, because neither alone is enough. `checkpoint_prefix` is what the operator
    typed and what the `.safetensors` files are named after — the only meaningful half —
    but it is not unique: three runs of one dataset here are all called `dh_bd_v1`, at
    two different ranks. The run directory is unique and is literally the folder on disk,
    so the pair always disambiguates. Older runs predate `checkpoint_prefix` and fall
    back to the directory alone rather than showing an empty name.
    """
    directory = point.output_dir.parent.name  # runs/train-<uuid>/checkpoints
    short = f"train-{directory.removeprefix('train-')[:8]}"
    name = str(point.config.get("checkpoint_prefix") or "").strip()
    return f"{name} · {short}" if name else short


def _render_fork_form(
    run: IngestionRun, fork_points: list[training_service.ResumePoint], *, busy: bool
) -> None:
    """Branch a checkpoint into an independent lineage with its own dataset.

    Unlike resume, this leaves the parent checkpoint untouched — every hyperparameter
    that shapes the learning-rate schedule stays pinned to the parent (see the
    caption below), so only the data intervention (images in/out, per-image weight)
    varies between siblings. That is what makes their loss curves comparable at all.
    """
    st.caption(
        "Continues from a checkpoint into a brand-new run — the parent stays intact "
        "and forkable again. Vary the dataset (add/remove images, weight some more "
        "or less) to measure its effect from a shared starting point."
    )

    if not fork_points:
        st.info(
            "Nothing to branch from yet. Launch a run from **New training** first — "
            "every checkpoint it exports becomes a point you can branch at, so a run "
            "with `save_every` 300 leaves you one every 300 micro-steps."
        )
        return

    with _db() as conn:
        fork_counts = {p.training_run_id: training_repo.count_forks_of(conn, p.training_run_id) for p in fork_points}

    # Two steps rather than one flat list. A dataset accumulates runs, and each run now
    # offers every step it exported — 44 points across 5 trainings on the dataset this
    # was built against, where the same step numbers (300, 600, 900…) recur in all of
    # them. Flat, "step 900" appears five times and identifies nothing.
    lineages: dict[Path, list[training_service.ResumePoint]] = {}
    for candidate in fork_points:
        lineages.setdefault(candidate.output_dir, []).append(candidate)

    output_dir = st.selectbox(
        "Training to fork from",
        list(lineages),
        format_func=lambda directory: _fork_lineage_label(lineages[directory][0]),
        key="fork_lineage_select",
    )
    if output_dir is None:
        return

    points = lineages[output_dir]
    already_forked = fork_counts[points[0].training_run_id]
    st.caption(
        f"{points[0].status} · {points[0].started_at:%Y-%m-%d %H:%M} · "
        f"{len(points)} forkable step{'s' if len(points) != 1 else ''}"
        + (f" · already forked {already_forked}×" if already_forked else "")
    )

    point = st.selectbox(
        "Checkpoint to fork",
        points,
        format_func=lambda p: (
            f"step {p.step} of {p.total_steps} · "
            + ("full state" if p.kind == "exact" else "weights only")
        ),
        # Keyed per lineage: a shared key would carry the previous training's selection
        # into a list that no longer contains it.
        key=f"fork_checkpoint_select_{output_dir.parent.name}",
    )
    if point is None:
        return

    if point.kind == "warm":
        # The trainer keeps one restorable state per run and rewrites it on every save,
        # so every step but the last survives only as a weights-only export. Saying this
        # here is the difference between a deliberate warm start and an operator
        # wondering why the branch's first hundred steps look unsettled.
        st.warning(
            f":material/whatshot: **Warm start.** Step {point.step} survives only as a "
            f"weights export, so this branch begins with a cold optimizer — no Adam "
            f"moments, no RNG stream, no sampler position. The weights themselves are "
            f"bit-identical to the parent's. Expect an unsettled first stretch while "
            f"Adam rebuilds its estimates.\n\n"
            f"Siblings forked from this same point are all equally cold, so comparing "
            f"them to **each other** stays valid — but do not read this branch against "
            f"the parent's own curve past step {point.step}. Create a control branch."
        )
    st.caption(
        f":material/info: Fingerprint warning in the branch's log is expected — it's "
        f"the new cache directory, not a corrupt checkpoint. Every branch pays a full "
        f"precache of its own dataset, control included."
    )

    label = st.text_input(
        "Branch label", value="", key="fork_branch_label",
        help="Used for the exported dataset folder and the checkpoint filename prefix.",
    )

    with st.container(horizontal=True):
        total_steps = st.number_input(
            "Total steps", min_value=point.step + 1,
            value=max(point.total_steps, point.step + 1), step=100,
            key="fork_total_steps",
        )
        save_every = st.number_input(
            "Save every", min_value=1, step=25,
            value=int(point.config.get("save_every") or 300),
            key="fork_save_every",
        )

    pinned = " · ".join(f"{name}={point.config.get(name)}" for name in _FORK_PINNED_LABELS)
    st.caption(
        f"Pinned to the parent — varying these would confound the data intervention: {pinned}"
    )

    samples = run.concept.samples
    images_tab, weights_tab = st.tabs(["Images", "Weights"])

    with images_tab:
        edited = st.data_editor(
            _fork_images_dataframe(samples),
            column_config={
                "sample_id": None,
                "incluir": st.column_config.CheckboxColumn("Include"),
                "imagen": st.column_config.TextColumn("Image", disabled=True),
                "tier": st.column_config.SelectboxColumn("Tier", options=list(Tier.__args__)),
                "caption": st.column_config.TextColumn("Caption", disabled=True),
            },
            hide_index=True,
            key="fork_images_editor",
            height=min(400, 40 + 35 * len(samples)),
        )

    with weights_tab:
        with st.container(horizontal=True):
            w_priority = st.number_input(
                "Priority weight", min_value=0.01, value=1.5, step=0.1, key="fork_w_priority"
            )
            w_good = st.number_input(
                "Good weight", min_value=0.01, value=1.0, step=0.1, key="fork_w_good"
            )
            w_bad = st.number_input(
                "Bad weight", min_value=0.01, value=0.5, step=0.1, key="fork_w_bad"
            )
        included_rows = edited[edited["incluir"]]
        counts = tier_counts(
            dict(zip(included_rows["sample_id"], included_rows["tier"])), len(included_rows)
        )
        st.caption(
            f"{counts['priority']} priority (×{w_priority}) · "
            f"{counts['good']} good (×{w_good}) · {counts['bad']} bad (×{w_bad})"
        )
        st.caption(
            ":material/warning: Weights are a per-image learning rate, not a filter — "
            "the batch is not renormalized, so down-weighting reduces an image's "
            "contribution without excluding it. The logged loss excludes the weight "
            "by design, so this chart will not visibly react to a weight-only change "
            "even though training does."
        )

    def _launch(*, as_control: bool) -> None:
        if not as_control and not label.strip():
            st.error("Branch label is required.")
            return
        branch_label = "control" if as_control else label.strip()
        dataset_name = f"{run.concept.concept_name}__{naming.slugify(branch_label)}"
        if dataset_name == run.concept.concept_name:
            st.error("Branch label must differ from the parent dataset.")
            return

        if as_control:
            selected_ids = {s.sample_id for s in samples if not s.is_excluded}
            tiers: dict[str, Tier] = {}
            profile = None
        else:
            included_rows_final = edited[edited["incluir"]]
            selected_ids = set(included_rows_final["sample_id"])
            tiers = dict(zip(included_rows_final["sample_id"], included_rows_final["tier"]))
            profile = WeightProfile(priority=w_priority, good=w_good, bad=w_bad)

        if not selected_ids:
            st.error("At least one image must be included.")
            return

        selected_samples = [
            s.model_copy(update={"is_excluded": False})
            for s in samples if s.sample_id in selected_ids
        ]
        content_hash = dataset_service.compute_content_hash(selected_samples)

        with st.spinner("Exporting branch dataset and pre-caching…"):
            export_result = export_service.export_branch_dataset(
                selected_samples, dataset_name, tiers=tiers or None, profile=profile
            )
            if tiers and not export_result.weights_are_effective:
                st.error(
                    "The chosen weights resolve to no effective change (all ×1.0) — "
                    "this branch would train identically to a control. Adjust the "
                    "weight profile or the tier assignment."
                )
                return

            with _db() as conn:
                try:
                    training_service.fork_training(
                        conn,
                        dataset_run_id=run.run_id,
                        fork_point=point,
                        branch=training_service.BranchSpec(
                            label=branch_label,
                            dataset_name=dataset_name,
                            trigger_word=run.concept.trigger_word,
                            dataset_content_hash=content_hash,
                            weight_profile=profile.model_dump() if profile else {},
                        ),
                        total_steps=int(total_steps),
                        save_every=int(save_every),
                    )
                except (
                    training_service.PrecacheFailed,
                    training_service.ForkUnavailable,
                ) as exc:
                    st.error(str(exc))
                    return
        _go_to_monitor(run.run_id)

    with st.container(horizontal=True):
        if st.button(
            "Create branch", icon=":material/call_split:", key="fork_create_branch",
            disabled=busy,
            help=_BUSY_HELP if busy else (
                "Use this when you're changing something — images included/excluded, "
                "or per-image weights (Images/Weights tabs above). "
                "¿Vas a cambiar algo (imágenes o pesos)? → Create branch"
            ),
        ):
            _launch(as_control=False)
        if st.button(
            "Create control branch", icon=":material/verified:", key="fork_create_control",
            disabled=busy,
            help=_BUSY_HELP if busy else (
                "Use this for the untouched baseline — same dataset, neutral weights, "
                "same starting checkpoint as its sibling — so you have something fair "
                "to compare a changed branch against. "
                "¿Necesitas una rama \"testigo\" sin cambios para comparar contra la "
                "anterior? → Create control branch"
            ),
        ):
            _launch(as_control=True)


def _monitor_run_label(run: training_repo.TrainingRun) -> str:
    """Name a run in the Monitor's picker, branches included."""
    directory = Path(str(run.config.get("output_dir") or "")).parent.name
    short = f"train-{directory.removeprefix('train-')[:8]}" if directory else "—"
    name = str(run.config.get("checkpoint_prefix") or "").strip()
    head = f"{name} · {short}" if name else short
    if run.branch_label:
        head = f"{head} · branch “{run.branch_label}”"
    when = f"{run.started_at:%Y-%m-%d %H:%M}" if run.started_at else "—"
    return f"{head} · {run.status} · {when}"


def _render_monitor(
    run: IngestionRun, train_runs: list[training_repo.TrainingRun]
) -> None:
    """Progress and log for one of this dataset's runs — live or finished.

    Being able to pick the run is what keeps a log readable after the process exits:
    this page used to show progress only while `status == "running"`, so the chart and
    log of a run vanished at the moment it became worth reading. It is also the only way
    to reach a branch's log without leaving the page.
    """
    st.caption(
        "Live progress and log output. Runs stay here after they finish, so a branch "
        "and its control can be read back side by side — the Metrics page overlays "
        "their curves."
    )

    if not train_runs:
        st.info("This dataset has no training runs yet.")
        return

    # One entry per launch, deliberately — unlike the fork picker, which collapses a
    # resumed lineage to one point. Two launches sharing an output_dir offer the same
    # checkpoint but wrote *different* log files, and reading either is the whole point
    # of this mode.
    running = next((r for r in train_runs if r.status == "running"), None)
    selected = st.selectbox(
        "Run",
        train_runs,
        index=train_runs.index(running) if running is not None else 0,
        format_func=_monitor_run_label,
        key=f"monitor_run_select_{run.run_id}",
    )
    if selected is None:
        return

    # Only a live run needs the 5s refresh; polling a finished run would re-read a log
    # and a CSV that cannot change again.
    if selected.status == "running":
        _render_live_run(selected.training_run_id)
    else:
        _render_run_body(selected)


@st.fragment(run_every="5s")
def _render_live_run(training_run_id: str) -> None:
    with _db() as conn:
        run = training_repo.get_training_run(conn, training_run_id)
        if run is None:
            return

        alive = training_runner.is_process_alive(run.pid)
        if not alive and run.status == "running":
            training_service.finalize_dead_run(conn, run, fallback_status="failed")
            run = training_repo.get_training_run(conn, training_run_id)
            if run is None:
                return

    _render_run_body(run)


def _render_run_body(run: training_repo.TrainingRun) -> None:
    status_line = f"Training · {run.status} · started {run.started_at}"
    if run.duration_seconds:
        status_line += f" · {run.duration_seconds / 60:.1f} min"
    if run.cost_estimate is not None:
        status_line += f" · ~${run.cost_estimate:.2f}"
    st.caption(status_line)
    if run.status == "failed" and run.error_message:
        st.caption(f"⚠ {run.error_message}")

    csv_path = training_service.training_log_csv_path(run)
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            if "step" in df.columns and "loss" in df.columns:
                st.line_chart(df.set_index("step")["loss"])
        except pd.errors.EmptyDataError:
            pass

    log_text = _tail_log(run.log_path)
    st.code(log_text or "Waiting for output…", language=None, height=240)

    if run.status == "running":
        if st.button("Stop training", icon=":material/stop:"):
            with _db() as conn:
                training_service.stop_training(conn, run.training_run_id)
            st.rerun()


def _tail_log(log_path: str) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    size = path.stat().st_size
    offset = max(0, size - LOG_TAIL_BYTES)
    text, _ = training_runner.read_log_tail(log_path, since_offset=offset)
    # The training loop's step progress prints "\r" + text with end="" — a
    # terminal-style overwrite-in-place bar. Redirected to a log file (as this
    # detached run's stdout is) there's no terminal to interpret that, so every
    # step's update just accumulates with no "\n" between them. splitlines()
    # treats bare "\r" as a line boundary same as "\n", so this recovers one
    # line per step instead of one line for the whole run.
    return "\n".join(text.splitlines())

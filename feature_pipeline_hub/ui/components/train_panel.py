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
from feature_pipeline.application import training_service
from feature_pipeline.domain import naming
from feature_pipeline.domain.models import IngestionRun
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


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    with _db() as conn:
        train_runs = training_repo.list_training_runs(conn, dataset_run_id=run.run_id)
    train_runs = [r for r in train_runs if r.kind == "train"]
    latest = train_runs[0] if train_runs else None  # newest first

    if latest is not None and latest.status == "running":
        _render_progress(latest.training_run_id)
        return

    _render_launch_form(run, latest)


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


def _render_launch_form(run: IngestionRun, latest: training_repo.TrainingRun | None) -> None:
    try:
        training_runner.resolve_environment()
    except training_runner.TrainingUnavailable as exc:
        st.caption(f":material/info: {exc}")
        return

    dataset_dir = training_service.dataset_dir_for(run.concept.concept_name)
    if not dataset_dir.is_dir() or not any(dataset_dir.iterdir()):
        st.caption(
            f":material/info: No exported dataset at training_runtime/datasets/"
            f"{run.concept.concept_name}/. Export it in Step 4 first."
        )
        return

    conflict = training_service.detect_dataset_version_conflict(run.concept.concept_name)
    dismissed_key = f"version_conflict_dismissed_{run.run_id}"
    if conflict is not None and not st.session_state.get(dismissed_key):
        _render_version_conflict_notice(conflict, dismissed_key)
        return

    if latest is not None:
        st.caption(f"Last run: {latest.status} · {latest.finished_at or 'in progress'}")

    with _db() as conn:
        resume_points = training_service.find_resume_points(conn, run.run_id)
    if resume_points:
        _render_resume_form(run, resume_points)
        st.markdown("**Start a new run**")

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

    if st.button("Start training", icon=":material/play_arrow:"):
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
        st.rerun()


def _render_resume_form(
    run: IngestionRun, resume_points: list[training_service.ResumePoint]
) -> None:
    """Offer to continue a previous run instead of starting over from step 0.

    Only `total_steps` is editable: the checkpoint's adapter was shaped by the
    original rank/alpha and would fail to load against different ones, so the
    rest of the hyperparameters are shown read-only rather than re-collected.
    """
    st.markdown("**Resume a previous run**")

    point = st.selectbox(
        "Checkpoint",
        resume_points,
        format_func=lambda p: (
            f"step {p.step} of {p.total_steps} · {p.status} · "
            f"{p.started_at:%Y-%m-%d %H:%M}"
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
            f"Resume from step {point.step}", icon=":material/fast_forward:"
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
        st.rerun()


@st.fragment(run_every="5s")
def _render_progress(training_run_id: str) -> None:
    with _db() as conn:
        run = training_repo.get_training_run(conn, training_run_id)
        if run is None:
            return

        alive = training_runner.is_process_alive(run.pid)
        if not alive and run.status == "running":
            training_service.finalize_dead_run(conn, run, fallback_status="failed")
            run = training_repo.get_training_run(conn, training_run_id)

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
                training_service.stop_training(conn, training_run_id)
            st.rerun()
    else:
        if st.button("Back to config"):
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

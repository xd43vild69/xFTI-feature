"""Settings panel component: Configure model paths, training environment, and tokens."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st

from feature_pipeline.infrastructure.app_settings import (
    AppSettings,
    default_config_path,
    load_app_settings,
    reset_app_settings,
    resolve_model_dir,
    resolve_training_python_path,
    resolve_training_runtime_dir,
    save_app_settings,
)
from feature_pipeline.infrastructure.model_prerequisites import check_model_status, get_saved_hf_token, save_hf_token


def render() -> None:
    st.markdown("## ⚙️ Application Settings & Model Storage")
    st.caption("Configure custom storage paths for model weights, training runtime, and environment variables.")

    settings = load_app_settings()

    # Active resolution preview
    krea2_active = resolve_model_dir("krea2")
    ltx23_active = resolve_model_dir("ltx23")
    runtime_active = resolve_training_runtime_dir()
    python_active = resolve_training_python_path()

    # Model status checks
    krea2_status = check_model_status("krea2", custom_dir=krea2_active)
    ltx23_status = check_model_status("ltx23", custom_dir=ltx23_active)

    # 1. Quick Presets / Detection
    _render_quick_presets()

    st.divider()

    # 2. Model Directories Section
    st.markdown("### 🧠 Base Model Weights")
    st.caption("Specify local directories containing pre-quantized base model weights.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Krea 2 (Krea-2-NF4)")
        krea_input = st.text_input(
            "Krea 2 Path",
            value=settings.krea2_model_dir or "",
            placeholder=str(runtime_active / "model"),
            help="Directorio con los pesos de Krea 2 (debe contener transformer, text_encoder, vae, etc.).",
            key="settings_krea2_input",
        )
        if krea2_status.is_ready:
            st.success(
                f":material/check_circle: **Listo** · `{krea2_active}` ({krea2_status.disk_size_gb:.1f} GB)"
            )
        else:
            st.warning(
                f":material/warning: **No listo** en `{krea2_active}`\n\n"
                f"Faltan: {', '.join(krea2_status.missing_items)}"
            )

    with col2:
        st.subheader("LTX 2.3 (LTX23-NF4)")
        ltx_input = st.text_input(
            "LTX 2.3 Path",
            value=settings.ltx23_model_dir or "",
            placeholder=str(runtime_active / "LTX23-NF4"),
            help="Directorio con los pesos de LTX 2.3 (debe contener transformer, text_encoder, connectors, etc.).",
            key="settings_ltx23_input",
        )
        if ltx23_status.is_ready:
            st.success(
                f":material/check_circle: **Listo** · `{ltx23_active}` ({ltx23_status.disk_size_gb:.1f} GB)"
            )
        else:
            st.warning(
                f":material/warning: **No listo** en `{ltx23_active}`\n\n"
                f"Faltan: {', '.join(ltx23_status.missing_items)}"
            )

    st.divider()

    # 3. Training Runtime Section
    st.markdown("### 🏋️ Training Runtime & Environment")
    col_rt1, col_rt2 = st.columns(2)

    with col_rt1:
        rt_input = st.text_input(
            "Training Runtime Directory",
            value=settings.training_runtime_dir or "",
            placeholder=str(Path(__file__).resolve().parents[3] / "training_runtime"),
            help="Directorio donde se guardan datasets exportados, runs y checkpoints.",
            key="settings_runtime_input",
        )
        if runtime_active.is_dir():
            # Check free space
            try:
                usage = shutil.disk_usage(runtime_active)
                free_gb = usage.free / (1024**3)
                st.info(f"📁 `{runtime_active}` ({free_gb:.1f} GB libres en disco)")
            except Exception:
                st.info(f"📁 `{runtime_active}`")
        else:
            st.warning(f"⚠️ El directorio `{runtime_active}` no existe aún (se creará al usarse).")

    with col_rt2:
        py_input = st.text_input(
            "Training Python Interpreter",
            value=settings.training_python_path or "",
            placeholder=str(runtime_active / "venv" / "bin" / "python"),
            help="Ruta al binario de Python en el entorno virtual dedicado para entrenamiento.",
            key="settings_python_input",
        )
        if python_active.is_file():
            try:
                py_ver = subprocess.check_output([str(python_active), "--version"], text=True).strip()
                st.success(f":material/check_circle: `{python_active}` ({py_ver})")
            except Exception:
                st.success(f":material/check_circle: `{python_active}`")
        else:
            st.warning(f"⚠️ No se encontró intérprete en `{python_active}`.")

    st.divider()

    # 4. Hugging Face Token Section
    st.markdown("### 🔑 API Tokens")
    current_token = settings.hf_token or get_saved_hf_token() or ""
    token_input = st.text_input(
        "Hugging Face Token (HF_TOKEN)",
        value=current_token,
        type="password",
        help="Token de Hugging Face requerido para descargar modelos base y checkpoints.",
        key="settings_hf_token_input",
    )

    st.divider()

    # 5. Actions: Save / Reset
    btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 4])

    with btn_col1:
        if st.button("💾 Guardar Configuración", type="primary", use_container_width=True):
            new_settings = AppSettings(
                krea2_model_dir=krea_input.strip() or None,
                ltx23_model_dir=ltx_input.strip() or None,
                training_runtime_dir=rt_input.strip() or None,
                training_python_path=py_input.strip() or None,
                hf_token=token_input.strip() or None,
            )
            save_app_settings(new_settings)

            if token_input.strip():
                save_hf_token(token_input.strip())

            st.success("¡Configuración guardada exitosamente!")
            st.rerun()

    with btn_col2:
        if st.button("🔄 Restablecer Predeterminados", use_container_width=True):
            reset_app_settings()
            st.info("Valores restablecidos a los predeterminados.")
            st.rerun()


def _render_quick_presets() -> None:
    """Render one-click quick configuration presets for known backup locations."""
    backup_root_linux = Path("/mnt/backup/ai-sandbox/trainer-files")
    backup_root_mac = Path("/Volumes/backup-1/ai-sandbox/trainer-files")

    detected_backup = None
    if backup_root_linux.is_dir():
        detected_backup = backup_root_linux
    elif backup_root_mac.is_dir():
        detected_backup = backup_root_mac

    if detected_backup is not None:
        krea_detected = detected_backup / "model"
        ltx_detected = detected_backup / "LTX23-NF4"

        with st.expander("⚡ Detección automática: Disco de backup detectado", expanded=True):
            st.write(
                f"Se detectó un directorio de backup con modelos en: **`{detected_backup}`**"
            )
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            with c1:
                st.caption(
                    f"• Krea 2: `{krea_detected}`\n\n"
                    f"• LTX 2.3: `{ltx_detected}`"
                )
            with c2:
                if st.button("Aplicar esta ruta", icon=":material/bolt:", use_container_width=True):
                    settings = load_app_settings()
                    settings.krea2_model_dir = str(krea_detected)
                    settings.ltx23_model_dir = str(ltx_detected)
                    save_app_settings(settings)
                    st.success("¡Rutas del backup aplicadas!")
                    st.rerun()

# Guía Teórica y Práctica de Diagnóstico de Entrenamiento LoRA en Krea 2
**Referencia Base: Plan Agosto (`vu_bd_v1` / `an_bd_v1`) y Caso de Estudio `lv_bd_v2` → `lv_bd_v4`**

Este documento establece el marco teórico, matemático y operativo para evaluar en tiempo real y post-mortem si un entrenamiento LoRA de personaje en **Krea 2 (Transformer 12B NF4 — Rectified Flow)** va por el camino correcto, cuándo está sub-entrenando y cuándo corre riesgo de quemar rasgos (overfitting).

---

## 1. Resumen Ejecutivo del Caso de Estudio

Durante la evolución del pipeline se presentaron dos anomalías opuestas respecto a la línea base histórica de **Agosto (`vu_bd_v1`, `an_bd_v1`, `dh_bd_v1`)**:

1. **Anomalía A — Sobre-entrenamiento destructivo y artefactos (`le_bd_v1` y `lv_bd_v2`, 21-Sep)**:
   * **Síntoma**: A los 1,500 pasos el personaje aparecía "quemado", con artefactos severos en piel/textura y pérdida de parecido facial general, a pesar de usar los mismos hiperparámetros de agosto.
   * **Causa raíz**:
     1. **`sampler: "legacy"` con múltiples aspect ratios**: En agosto, los datasets tenían todos sus recortes agrupados en 1 a 4 *buckets* de igual tamaño (`dh_bd_v1`: 1 bucket `64x64`; `vu_bd_v1`: 4 buckets balanceados). En septiembre, tras introducir recortes/transformaciones variables, `lv_bd_v2` y `le_bd_v1` generaron **10 y 11 buckets**, donde entre 5 y 10 buckets contenían **1 sola imagen**. Como `legacy` elegía primero un bucket al azar con probabilidad uniforme ($1/11$) y luego una imagen dentro de él, las imágenes aisladas recibieron **~135 pasadas (épocas)** mientras que el bucket principal (`64x64`, con 23 imágenes) recibió apenas **22 pasos en total (< 1 época por imagen)**.
     2. **Caché huérfana (`prune_orphans: "warn"`)**: En `le_bd_v1` permanecían 52 imágenes antiguas (`.pt`) en `cache/le_bd_v1` de sesiones previas de renombrado/curaduría, entrenando 85 imágenes en lugar de las 33 activas.

2. **Anomalía B — Sub-entrenamiento severo / Pérdida total de parecido (`lv_bd_v3`, 22-Sep)**:
   * **Síntoma**: Al corregir el `sampler` a `"epoch"`, pero cambiando simultáneamente `optimizer_eps` (`1e-8` $\rightarrow$ `1e-6`), `lr` (`1.5e-4` $\rightarrow$ `1.0e-4`), `lora_rank` (`32` $\rightarrow$ `16`) y activando `use_ema: true` (`0.99`), el modelo en los pasos 1,000 y 1,500 no tenía parecido alguno con la modelo.
   * **Causa raíz**:
     1. **Estrangulamiento de AdamW por `optimizer_eps: 1e-6`**: En el Transformer 12B de Krea 2 hay **58,652,880 parámetros entrenables** distribuidos en 264 capas. Con una norma global de gradiente `gnorm ≈ 0.010`, el gradiente individual por parámetro $|g_i|$ en los `transformer_blocks` es del orden de **$8 \times 10^{-8}$ (`8e-8`)**. En la actualización de AdamW:
        $$\Delta w_i = -\text{lr} \cdot \frac{\hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$
        Al subir $\epsilon$ de `1e-8` a `1e-6` ($100\times$ mayor), el término $\epsilon = 10^{-6}$ pasó a ser **12.5 veces mayor que el propio gradiente ($\sqrt{\hat{v}_i} \approx 8 \times 10^{-8}$)**, dividiendo silenciosamente el *learning rate* efectivo por **~12.5x** (y junto con la bajada de `lr` y `EMA`, redujo la magnitud de `lora_B` en **17.5x**).
     2. **Horizonte de `EMA (0.99)` en corridas de 375 updates**: Con `1500` pasos y `grad_accum_steps = 4`, toda la corrida consta de sólo **375 actualizaciones** del optimizador. Un `ema_decay = 0.99` tiene una ventana de $\frac{1}{1 - 0.99} = 100$ updates (`400` pasos), arrastrando los pesos exportados hacia la inicialización en cero.

3. **Resolución y Convergencia (`lv_bd_v4`, 22-Sep)**:
   * Se restauró **íntegramente la receta matemática de Agosto (`vu_bd_v1`)** (`lr: 0.00015`, `lora_rank: 32`, `lora_alpha: 16`, `optimizer_eps: 1e-8`, `optimizer_betas: [0.9, 0.999]`, `timestep_weighting: "none"`, `use_ema: false`, `lora_dtype: "bf16"`), aplicando **únicamente** las dos correcciones estructurales:
     * `"sampler": "epoch"` (garantiza que cada una de las 34 imágenes se entrene exactamente 1 vez por época, sin sesgo de aspect ratio).
     * `"prune_orphans": "delete"` y filtrado por `cache_manifest.json` (garantiza que jamás se entrenen tensores huérfanos).

---

## 2. Tabla Comparativa Maestra: Agosto (Plan Base) vs Casos Analizados

| Métrica / Parámetro | Plan Base Agosto (`vu_bd_v1` / `an_bd_v1`) | Fallo A: Sobre-entrenado (`lv_bd_v2` / `le_bd_v1`) | Fallo B: Sub-entrenado (`lv_bd_v3`) | Solución Óptima (`lv_bd_v4`) |
| :--- | :---: | :---: | :---: | :---: |
| **Estado del Parecido** | ✅ **Excelente (Referencia)** | ❌ **Quemado / Artefactos** | ❌ **Sin parecido (Vacío)** | ✅ **Punto Dulce Recuperado** |
| `lr` / `lora_rank` / `alpha` | `1.5e-4` / `32` / `16` | `1.5e-4` / `32` / `16` | `1.0e-4` / `16` / `16` | **`1.5e-4` / `32` / `16`** |
| `optimizer_eps` | **`1e-08`** | `1e-08` | `1e-06` *(bloqueó AdamW)* | **`1e-08`** |
| `sampler` | `legacy` *(1–4 buckets iguales)* | `legacy` *(10–11 buckets desiguales)* | `epoch` | **`epoch`** *(10 buckets equitativos)* |
| `use_ema` | `false` | `false` | `true (0.99)` | **`false`** |
| **Pasadas por imagen (1500 steps)** | Uniforme (`~39–44x`) | Sesgado (`0.8x` vs `135x`) | Uniforme (`44.1x`) | **Uniforme (`44.1x` exactas)** |
| **`mean_loss_head` (Inicio 50 upd)** | `0.1251 – 0.1362` | `0.1588 – 0.1684` | `0.1471` | **`0.1595`** |
| **`mean_loss_tail` (Final 50 upd)** | **`0.0669 – 0.0959`** | `0.1221 – 0.1241` | `0.1300` *(estancado)* | **`0.1078`** *(en paso 1000)* |
| **Caída `loss_delta` (`tail - head`)** | **`-0.0402 a -0.0536`** | `-0.0367 a -0.0443` | `-0.0171` *(insuficiente)* | **`-0.0517`** *(en paso 1000)* |
| **`grad_norm` mediano (`p50`)** | **`0.0165 – 0.0186`** | `0.0121 – 0.0168` | `0.0148` *(sin paso efectivo)* | **`0.0112 – 0.0175`** |
| **Norma `‖lora_B‖` en Step 250** | `~0.19` | `~0.20` | `0.035` | **`0.1931`** |
| **Norma `‖lora_B‖` en Step 500** | `~0.78` | `~0.81` | `0.072` | **`0.7810`** |
| **Norma `‖lora_B‖` en Step 750** | `~1.12` | `~1.15` | `0.1049` | **`1.1334`** |
| **Norma `‖lora_B‖` en Step 1000** | **`~1.26 – 1.32`** | `~1.34` *(mal distribuido)* | `0.1240` *(10.5x menor)* | **`1.2850`** ⭐ |
| **Norma `‖lora_B‖` en Step 1250** | **`~1.32 – 1.38`** | `~1.39` *(mal distribuido)* | `0.1310` | **`1.3329`** ⭐ |

---

## 3. Base Teórica: Las 4 Leyes para Medir un Buen Entrenamiento en Krea 2

Para diagnosticar cualquier entrenamiento futuro en la interfaz (**Step 4 · Train → Monitor**) o desde `train_log.csv`, utiliza estas **4 métricas fundamentales**:

### Ley 1: La Curva de Pérdida (`head_loss` → `tail_loss` y `Δ Loss`)
En modelos *Rectified Flow* (como Krea 2), el objetivo predice la velocidad $v = \epsilon - x_0$ interpolada por un tiempo aleatorio $\sigma \in (0, 1)$:
$$x_\sigma = (1 - \sigma) x_0 + \sigma \epsilon$$
* **Por qué un solo paso engaña**: Cuando el muestreador elige $\sigma > 0.60$ (mucho ruido), la pérdida instantánea (`loss_step`) es naturalmente baja (`0.04 – 0.09`). Cuando elige $\sigma < 0.05$ (casi imagen limpia), la pérdida instantánea salta a `0.50 – 1.08`.
* **Cómo medirla correctamente**: Nunca evalúes un paso aislado; evalúa siempre la **ventana móvil de 50 updates (`200` pasos)** (`loss_tendencia_50` en el gráfico y `Loss Reciente (Tail)` en la tarjeta):
  1. **Caída saludable (`Δ Loss`)**: Entre el inicio (`head_loss`, primeros 50 updates) y la zona óptima (pasos `1,000 – 1,350`), la pérdida media debe caer entre **`-0.0400` y `-0.0550` puntos** (una reducción del **30% al 40%** respecto al valor inicial).
  2. **Banda objetivo absoluta (`tail_loss`)**: Debe descender desde `~0.140–0.160` hasta situarse en la banda **`0.0670 – 0.1050`** (línea rosada de referencia `0.0880`).
  3. **Señal de alarma**:
     * Si en el paso `1,000` la caída `Δ Loss` es menor a `-0.0200` (ej. `-0.0171` en `lv_bd_v3`), **el modelo no está aprendiendo** (revisar `optimizer_eps` o `lr`).
     * Si `tail_loss` cae por debajo de **`0.0550`** de forma sostenida, **el modelo está memorizando píxeles y fondos (overfitting)**.

### Ley 2: La Evolución Física de los Pesos (`mean ‖lora_B‖₂`)
En PEFT LoRA, la matriz $A$ se inicializa con distribución Kaiming uniforme y la matriz $B$ se inicializa en **cero exacto ($B = 0$)** para que en el paso 0 el adaptador no altere el modelo base ($\Delta W = \frac{\alpha}{r} B A = 0$).
Por tanto, la norma L2 media de todas las matrices `lora_B` del archivo `.safetensors` mide **cuánta energía real ha acumulado el LoRA**:

* **Paso 250** *(fin del warmup inicial)*: `mean ‖lora_B‖` debe rondar **`0.18 – 0.25`**.
* **Paso 500** *(mitad de fase de consolidación)*: `mean ‖lora_B‖` debe subir a **`0.70 – 0.85`**.
* **Paso 750** *(inicio de zona de parecido)*: `mean ‖lora_B‖` debe situarse en **`1.05 – 1.18`**.
* **Paso 1,000 – 1,250** *(Sweet Spot / Punto Dulce)*: `mean ‖lora_B‖` debe alcanzar **`1.25 – 1.36`**.
* **Paso 1,500**: `mean ‖lora_B‖` estabiliza en **`1.36 – 1.44`**.

> **Comando rápido para verificar la fuerza de cualquier checkpoint `.safetensors` en el servidor**:
> ```bash
> training_runtime/venv/bin/python -c '
> from safetensors.torch import load_file
> sd = load_file("RUTA_AL_CHECKPOINT.safetensors")
> b = [v.float().norm().item() for k, v in sd.items() if "lora_B" in k]
> print(f"mean lora_B norm = {sum(b)/len(b):.5f}")
> '
> ```

### Ley 3: Dinámica de `Grad Norm` (`grad_norm_p50`)
* **Fase Warmup (`Paso 1 → 200`)**: Como $B = 0$ al inicio y $\nabla A = B^\top \nabla y$, al principio sólo $B$ recibe gradiente mientras el `lr` sube en rampa. Es completamente normal que `grad_norm` empiece en **`0.0040 – 0.0090`**.
* **Fase Activa (`Paso 250 → 1,000`)**: Una vez que $B$ crece, $A$ y $B$ reciben gradiente simultáneamente. El `grad_norm` mediano debe situarse entre **`0.0100` y `0.0250`** (referencia Agosto: `0.0175`).
* **Fase de Meseta (`Paso 1,000 → 1,500`)**: El `grad_norm` se estabiliza suavemente alrededor de **`0.0100 – 0.0150`** mientras el coseno reduce el `lr`.

### Ley 4: Épocas Efectivas y Simetría de Muestreo (`sampler: "epoch"`)
Para un dataset curado de retratos/cuerpo de **$N$ imágenes** (típicamente $N \in [25, 40]$):
$$\text{Épocas Efectivas} = \frac{\text{Pasos Ejecutados}}{N}$$
* **Zona Roja (`< 15 épocas`, ej. Paso `< 500` con 34 fotos)**: Sólo estructura general e iluminación; el rostro aún es genérico.
* **Zona Naranja (`15 a 25 épocas`, ej. Paso `500 – 850`)**: El rostro empieza a parecerse, pero aún puede fallar en ángulos laterales o planos abiertos.
* **Zona Verde — Sweet Spot (`28 a 38 épocas`, ej. Paso `950 – 1,300` con 34 fotos)**: **Máxima fidelidad de identidad facial y corporal** conservando total plasticidad para cambiar ropa, poses, fondos e iluminación (y para animar luego en video).
* **Zona Púrpura (`> 42 épocas`, ej. Paso `> 1,450`)**: Rasgos muy fijados; evaluar siempre contra los checkpoints de los pasos `1,000` y `1,250` para descartar rigidez.

---

## 4. Guía de Lectura en Vivo de las 3 Etapas del Monitor UI

Tal como comprobamos en la corrida `lv_bd_v4`:

1. **Al inicio (`Paso ~35 – 150`, `1 a 4 épocas`)**:
   * El indicador muestra **`⏳ CALIBRANDO VENTANA INICIAL (Warmup)`** y `⏳ Rampa Warmup` en `grad_norm`.
   * La línea celeste oscila mientras reúne las primeras 50 muestras de `sigma`. No tomes decisiones antes del paso 400.
2. **A mitad de camino (`Paso ~400 – 700`, `12 a 20 épocas`)**:
   * El *warmup* de 100 updates (`400` pasos) acaba de terminar y el `lr` está al máximo (`1.50e-4`).
   * Es normal ver varios picos celestes altos (`0.60 – 1.05`) cuando el generador muestrea `sigma < 0.03`, los cuales mantienen temporalmente la media móvil (`loss_tendencia_50`) alrededor de `0.145 – 0.155`.
   * Lo que debes verificar aquí es que el indicador inferior diga **`📉 Descendiendo`** y que el `Grad Norm` marque **`✓ Óptimo`**.
3. **Al entrar al Punto Dulce (`Paso ~950 – 1,250`, `28 a 37 épocas`)**:
   * La curva azul oscura (`loss_tendencia_50`) baja hasta `~0.095 – 0.108`, pegándose a la línea rosada (`0.088`).
   * La barra se pone en **🟢 EN ZONA DE ÉXITO DE AGOSTO (`95% – 100%`)** con una caída `Δ Loss` de **`~ -0.045 a -0.052`**.
   * Los checkpoints **`step_1000`** y **`step_1250`** son tus candidatos principales de producción.

---

## 5. Receta Maestra Oficial Krea 2 (Base Agosto Corregida)

Configuración canónica validada para datasets de **25 a 40 imágenes**:

```json
{
  "total_steps": 1500,
  "lr": 0.00015,
  "lora_rank": 32,
  "lora_alpha": 16,
  "batch_size": 1,
  "grad_accum_steps": 4,
  "save_every": 250,
  "seed": 42,
  "warmup_steps": 100,
  "warmup_units": "updates",
  "lr_scheduler": "cosine",
  "lr_num_cycles": 3,
  "min_lr_ratio": 0.1,
  "sampler": "epoch",
  "lora_dtype": "bf16",
  "high_precision_targets": false,
  "lora_target": "all",
  "optimizer_betas": [0.9, 0.999],
  "optimizer_eps": 1e-08,
  "timestep_weighting": "none",
  "content_or_style": "balanced",
  "noise_offset": 0.0,
  "caption_dropout_rate": 0.05,
  "use_ema": false,
  "ema_decay": 0.99
}
```

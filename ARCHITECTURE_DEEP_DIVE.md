# Feature Pipeline Hub — Deep Dive Técnico

Documento de referencia que explica el sistema `xFTI-feature` de extremo a extremo,
estructurado en tres pilares: **Lógica de negocio**, **Stack tecnológico** y
**Fundamentos matemáticos**. Cada afirmación está anclada a un archivo/línea real del
repositorio — no es una descripción aspiracional, es lo que el código hace hoy.

Alcance: `feature_pipeline_hub/` (el pipeline de curaduría + entrenamiento) y
`agent/` (el MCP Host que lo automatiza). No cubre `training_runtime/` como tal
(es un artefacto generado por `scripts/setup_training_runtime.sh`, no código fuente).

---

## 1. Arquitectura y Lógica de Negocio

### 1.1 Clean Architecture: aislamiento real, no solo carpetas

La estructura bajo `src/feature_pipeline/` no es una convención cosmética — cada capa
tiene una regla de dependencia que el propio código impone:

```
domain/          → Pydantic puro. Cero I/O, cero imports de infrastructure/ui.
application/     → Lógica de negocio. Depende de domain + infrastructure interfaces.
infrastructure/  → SQLite, filesystem, subprocess. Implementa lo que application usa.
ui/              → Streamlit. Es el ÚNICO consumidor autorizado de application vía state.py.
workers/         → Procesos standalone en OTRO intérprete Python. No importan nada de src/.
```

**`domain/models.py`** define el vocabulario del sistema en Pydantic puro:
`DatasetSample`, `ImageMetrics`, `ConceptGroup`, `IngestionRun`, `DatasetManifest`.
Ningún modelo de dominio abre un archivo, hace una query SQL o importa Streamlit —
son estructuras de datos validadas y funciones puras sobre ellas
(`domain/validators.py`, `domain/cost.py`). Esto es lo que permite testear reglas de
negocio (p.ej. "un `concept_name` no puede contener `/`" en
[`domain/models.py:65-72`](feature_pipeline_hub/src/feature_pipeline/domain/models.py))
sin tocar disco ni base de datos.

**`application/`** es donde vive la lógica real, un módulo por paso del pipeline
(`dataset_service`, `quality_service`, `image_service`, `export_service`,
`training_service`, `recaption_service`, `caption_service`). La disciplina aquí es
notable: `quality_service.py` lo dice explícito en su docstring — *"Everything here is
pure: it reads already-computed sample metrics and returns plain values, so the UI can
recompute freely without touching disk."* Todas las funciones de clustering,
estadísticas de captions, distribución de aspect ratios, etc., son funciones puras
`list[DatasetSample] -> valor`. Ni una sola abre un archivo.

**`infrastructure/`** es la única capa que toca el mundo exterior: SQLite
(`database.py`, `ingestion_repository.py`, `training_repository.py`,
`version_repository.py`), filesystem (`storage.py`, `hf_exporter.py`), y lanzamiento de
subprocesos (`recaption_runner.py`, `training_runner.py`). `application/` depende de
estas piezas por composición (les pasa una `sqlite3.Connection` ya abierta), nunca al
revés.

**`ui/`** es intencionalmente delgada. `ui/state.py` es, según el propio `CLAUDE.md`,
*"the sole bridge between UI session state and the SQLite-backed application layer"* —
toda lectura/escritura de la base de datos desde la UI pasa por ahí, nunca directo
desde un componente. `ui/steps/*.py` son wrappers finos que registran páginas de
Streamlit; `ui/components/*.py` contienen la lógica real de cada panel, pero siempre
delegando a `application/` a través de `state.py`.

El resultado práctico de este aislamiento: `mcp_server/server.py` (construido después,
para exponer el pipeline a agentes) pudo reutilizar el 100% de `application/` sin tocar
una línea de `ui/`. El propio comentario del server lo confirma:

```python
# mcp_server/server.py
# "Every tool opens its own SQLite connection and closes it before returning,
#  matching the per-operation-connection convention documented in ui/state.py"
```

Dos consumidores completamente distintos (Streamlit con estado de sesión por
navegador, y un servidor stdio sin estado) comparten la misma capa de aplicación
porque esa capa nunca supo que Streamlit existía.

### 1.2 Los 5 pasos y el Feature Store lógico

El flujo de trabajo (Import → Curate → Quality → Export → Train) no es un wizard
lineal con estado en memoria — es una máquina de estados persistida en SQLite,
donde cada paso lee y escribe contra las mismas tablas.

**Modelo de datos (`infrastructure/database.py`):**

```
concepts          → un dataset con nombre (concept_name + trigger_word)
ingestion_runs     → UNA importación de ese concept. Re-escanear crea un run NUEVO,
                     nunca sobreescribe uno viejo.
samples             → pertenecen a un run: caption, ImageMetrics, flags de validación
dataset_versions    → snapshots de export, con manifest_json para diffing
training_runs       → cada subprocess lanzado (precache/train), con pid, log_path,
                     status y telemetría
```

La decisión de diseño más importante aquí es que **`run_id`, no `concept_id`, es lo
que la UI selecciona**. Esto significa que puedes re-importar el mismo concept diez
veces (por ejemplo, tras añadir más fotos a la carpeta fuente) y cada importación
queda como un run independiente y comparable — nunca pierdes el estado de curaduría
de una versión anterior por error.

**Paso 1 — Import** (`dataset_service.create_ingestion_run`): escanea una carpeta,
calcula `ImageMetrics` para cada imagen (ver §3.1), asocia `.txt` como caption si
existe, y persiste todo como un `IngestionRun` nuevo vía
`ingestion_repository.save_ingestion_run`.

**Paso 2 — Curate**: edición manual de captions y flags (`is_excluded`, `is_flagged`),
más recaption asistido por IA (§2.2). Los widgets de edición de caption son
**versionados** — `caption_widget_key` / `CAPTION_VERSIONS_KEY` en `ui/state.py` —
porque un widget de Streamlit con `key` fijo ignora un nuevo `value=` en el rerun; el
contador de versión fuerza a Streamlit a remontar el widget cuando el caption cambia
desde fuera (batch replace, IA, edición rápida en el panel de calidad).

**Paso 3 — Quality** (`application/quality_service.py`): deduplicación perceptual
(§3.1), detección de captions vacíos, estadísticas de longitud de caption vs. límite
del text encoder, y ranking de nitidez. Todo de solo lectura sobre `samples` — no
muta nada, solo informa qué flags debería poner el usuario en el paso 2.

**Paso 4 — Export** (`application/export_service.py` +
`infrastructure/hf_exporter.py`): materializa los samples activos (`not is_excluded`)
como una carpeta plana `training_runtime/datasets/<destination_name>/` — el formato
exacto que el Paso 5 espera. Cada export genera un `DatasetManifest`
(`domain/models.py:134`) con un `content_hash` — ver §1.2.1 — para poder responder
"¿esta v2 realmente cambió algo respecto a v1?" sin recalcular todo desde cero.

**Paso 5 — Train** (`application/training_service.py`): orquesta precache + train
como dos subprocesos separados en un intérprete distinto. Ver §1.3 y §3.3.

#### 1.2.1 El fingerprint de contenido

`dataset_service.compute_content_hash` (línea 186) resuelve un problema real: dos
exports del mismo run pueden diferir en bytes de archivo (recompresión, metadata EXIF)
sin que el *contenido curado* haya cambiado. La función hashea el par ordenado
`(phash, caption)` de cada sample activo:

```python
active = sorted((s.metrics.phash, s.caption) for s in samples if not s.is_excluded)
digest = hashlib.sha256()
for phash, caption in active:
    digest.update(phash.encode("utf-8"))
    digest.update(b"\x00")
    ...
```

Ordenar antes de hashear hace que el resultado sea independiente del orden de escaneo
del filesystem. Usar `phash` en vez del hash criptográfico del archivo hace que el
fingerprint tolere cambios que un humano consideraría "el mismo dataset" (re-encode a
otro formato) mientras detecta los que sí importan (una imagen añadida/quitada, un
caption editado).

### 1.3 Orquestación asíncrona: dos fronteras de proceso, dos contratos distintos

El sistema tiene **dos patrones de subprocess completamente distintos**, elegidos
deliberadamente según la duración del trabajo — mezclarlos sería un error, y el código
los mantiene separados en dos módulos de infraestructura distintos.

**`recaption_runner.py` — corto, streaming.** El recaption con Qwen3-VL tarda
segundos por lote (~2s/imagen en GPU). El runner usa `subprocess.Popen` con pipes
(`stdin=PIPE, stdout=PIPE`), escribe el job como JSON a stdin, y hace `yield` de cada
línea JSON que el worker imprime a stdout **mientras el proceso corre**:

```python
# infrastructure/recaption_runner.py
for line in process.stdout:
    ...
    yield json.loads(line)
process.wait()
```

Es una llamada bloqueante desde la perspectiva del caller (la UI espera a que termine
el lote), pero permite mostrar progreso incremental porque el generador entrega cada
evento tan pronto como llega.

**`training_runner.py` — largo, detached.** Pre-cache y train pueden tardar de
minutos (`PRECACHE_TIMEOUT_SECONDS = 20 * 60` en
[`training_service.py:30`](feature_pipeline_hub/src/feature_pipeline/application/training_service.py))
a horas. Aquí el patrón es opuesto: `start_new_session=True` desvincula el subproceso
del árbol de procesos de Streamlit —*"survives the parent (Streamlit) exiting"*, dice
el comentario en la línea 118 — la salida se redirige a un archivo de log, y
`launch()` devuelve `(pid, log_path)` **inmediatamente**, sin esperar nada:

```python
process = subprocess.Popen(
    [str(env.python), "-u", str(script)],
    ...
    stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True,
)
return process.pid, str(log_path)
```

El progreso se recupera después — de cualquier rerun de Streamlit, cualquier sesión de
navegador nueva, o una llamada MCP — tallando el archivo de log
(`read_log_tail`, que lee solo los bytes nuevos desde el último offset leído) y
consultando la fila `training_runs` en SQLite. Nada se mantiene en memoria del proceso
padre.

**Detección de vida sin zombies.** `is_process_alive(pid)` (línea 181) no es un simple
`os.kill(pid, 0)`. Un hijo detached que terminó pero nunca fue `wait()`-eado queda como
zombie (`state == "Z"` en `/proc/<pid>/stat`) y seguiría respondiendo "vivo" a
`os.kill(pid, 0)` para siempre — lo que dejaría el lock de GPU
(`training_service.is_training_active`) bloqueado eternamente tras un run terminado.
La función lee el estado real del proceso en `/proc` y, si es zombie, intenta
reaparlo con `os.waitpid(pid, os.WNOHANG)`.

**Parada elegante escalada.** `stop_process` (línea 214) manda `SIGINT` primero —
*"the training loop's own KeyboardInterrupt handling is what writes a resumable
checkpoint before exiting"*— y solo escala a `SIGTERM` si el proceso no responde en
10 segundos. Esto es lo que permite parar un entrenamiento de horas sin perder el
checkpoint.

**El contrato precache → train.** `training_service.start_training` (línea 92)
encadena ambas fases en una sola llamada: precache corre **bloqueante** con timeout
(`_run_precache_blocking`), y solo si termina con éxito se lanza train detached
(`_launch_train`). Pero el MCP server (que no puede bloquear una tool call por 20
minutos) usa el split explícito: `launch_precache` (fire-and-forget) +
`precache_status` (poll) + `launch_train` — mismo subprocess launch subyacente, sin
encadenarlos en una sola llamada síncrona.

**Los workers no saben que los observan.** `workers/precache_worker.py` y
`workers/train_worker.py` están portados desde el proyecto upstream LoRAlab y deben
permanecer diffables contra el original — no se les añade instrumentación por dentro.
`precache_worker.py` sigue siendo byte a byte idéntico; `train_worker.py` conserva la
**lógica** exacta del upstream pero sus docstrings se reescribieron aquí (sólo en
inglés, uno por función no obvia), así que se compara por AST, no por bytes. `workers/_telemetry.py` envuelve sus entrypoints `__main__`
**desde afuera** para emitir eventos JSON-lines de ciclo de vida
(`worker_started`, `worker_finished`, `worker_failed`) sin tocar el código vendorizado.
`training_runner.read_lifecycle_event` lee únicamente los últimos ~4KB del log para
encontrar esa línea final barato, incluso en logs de horas de duración.

---

## 2. Stack Tecnológico y Protocolos

### 2.1 Base: Python, Streamlit, Pydantic, SQLite

**Python 3.11+** (`requires-python = ">=3.11"` en `pyproject.toml`), tipado
estrictamente con `mypy --strict` sobre `src/feature_pipeline` y `mcp_server`
(deliberadamente excluidos: `ui/` porque el tipado de Streamlit es ruidoso, y
`workers/` porque deben permanecer diffables contra el upstream).

**Pydantic ≥2.7** no es solo para (de)serialización — es la capa de validación de
esquema del dominio. Cada modelo en `domain/models.py` encierra invariantes que de
otro modo vivirían dispersas: `ImageMetrics.width: int = Field(gt=0)`,
`ConceptGroup.concept_name` con un `field_validator` que rechaza path traversal
(`"/"`, `"\\"`, `"."`, `".."`) porque ese string se usa literalmente como nombre de
carpeta en disco. `training_service.TrainingConfig` usa `model_config =
ConfigDict(frozen=True)` con `Field(gt=0)` en cada hiperparámetro — un `total_steps=0`
o un `lr` negativo se rechazan en el momento en que se *construye* el objeto, no
cuando el training worker ya está corriendo y desperdicia GPU.

**Streamlit ≥1.35** maneja la UI y el estado de sesión, pero el propio `CLAUDE.md`
documenta la disciplina que lo mantiene desacoplado del resto: `ui/state.py` como
único puente hacia SQLite, y el dashboard de monitoreo de training usa
`@st.fragment(run_every="5s")` para autorefrescarse sin re-ejecutar la página
completa — necesario porque `training_runner` es fire-and-forget y el único modo de
saber el progreso es tallar el log periódicamente.

**SQLite** es la base de persistencia y tracking de ejecuciones, con una regla de
conexión no negociable documentada en varios sitios del código: **las conexiones se
abren por operación y nunca se cachean**, porque los reruns de Streamlit pueden
aterrizar en threads distintos y `sqlite3.Connection` no es segura para compartir
entre threads. El mismo patrón (`get_connection()` → usar → `close()`) se repite en
`ui/state.py`, `mcp_server/server.py` (`_db()` context manager) y
`ui/step_telemetry.py`. Las migraciones de esquema se hacen vía diccionarios
`*_COLUMN_MIGRATIONS` con `ALTER TABLE ... ADD COLUMN`, nunca editando el
`CREATE TABLE IF NOT EXISTS` en sitio — eso dejaría intactas las bases de datos ya
existentes.

### 2.2 IA: Qwen3-VL-4B para recaptioning, ecosistema HF para entrenamiento

**Qwen3-VL-4B-Instruct** es el modelo de recaptioning automático
(`workers/caption_qwen3vl.py`), y no es una elección arbitraria: es *el mismo modelo*
que Krea 2 usa como text encoder, cargado desde
`training_runtime/model/text_encoder/`. La ventaja práctica: no hay que descargar ni
mantener un segundo modelo — el peso ya está en disco para el pipeline de
entrenamiento, y el recaptioning simplemente lo reutiliza en modo generación de texto
(vía `transformers.Qwen3VLForConditionalGeneration`) en lugar de modo embeddings (vía
`diffusers`). El checkpoint tiene `tie_word_embeddings: true`, así que aunque no trae
`lm_head.weight` explícito, `model.tie_weights()` lo materializa desde
`embed_tokens.weight` — la generación es real, no un truco.

Dos prompts fijos controlan el estilo del caption
(`CAPTION_INSTRUCTION` / `DETAILED_CAPTION_INSTRUCTION`, ambos verbatim de Fizgig):
uno conciso de una frase, otro detallado de 2-4 frases cubriendo pose, ropa, punto de
vista de cámara, y si el rostro es visible. Ambos instruyen explícitamente *"State
only what is visible — no speculation, no names, no style commentary"* — una
restricción deliberada contra alucinaciones del modelo de visión.

**El ecosistema Hugging Face** (`torch`, `diffusers`, `peft`, `accelerate`, más
`bitsandbytes` para los optimizadores 8-bit) vive exclusivamente en
`training_runtime/venv/` — un intérprete Python **separado** del que corre
`feature_pipeline_hub` (ver §1.3). `peft.LoraConfig` y `peft.get_peft_model` son los
que implementan la adaptación LoRA en `train_worker.py` (detalle matemático en §3.3);
`accelerate` gestiona la colocación en GPU y el training loop de precisión mixta.

### 2.3 Model Context Protocol: exponer el pipeline a agentes autónomos

`mcp_server/server.py` usa `FastMCP` (transporte stdio) para envolver
`application/` como 11 tools invocables por un agente:
`list_dataset_runs`, `get_dataset_health`, `get_run_detail`, `import_dataset`,
`revalidate_run`, `export_dataset`, `quality_summary`, `start_lora_training`,
`continue_lora_training`, `get_training_status`, `stop_training`.

Tres decisiones de diseño hacen que esto funcione como una capa fina y no como una
reimplementación:

1. **Cada tool es un wrapper delgado.** Ninguna tool contiene lógica de negocio
   propia — todas abren una conexión (`_db()`), llaman a una función de
   `application/`, y devuelven `model_dump(mode="json")`. La lógica ya existía; el
   servidor solo la hace direccionable.

2. **Sin estado entre llamadas, por diseño.** El servidor no tiene memoria de sesión
   — cada tool call es una conexión SQLite abierta y cerrada. Esto es lo que
   obliga al contrato explícito entre `start_lora_training` y `continue_lora_training`:
   los hiperparámetros **no se persisten** entre ambas llamadas
   (`start_lora_training`'s docstring: *"call continue_lora_training with the SAME
   hyperparameters"*) — un agente que no lea el docstring puede lanzar el precache
   con un `lora_rank=16` y continuar el train con el default `lora_rank=16` sin
   darse cuenta de que si hubiera pasado un valor distinto en la primera llamada,
   el mismatch habría entrenado en silencio con la config equivocada. Es el punto
   de mayor riesgo de toda la superficie de la API, y por eso el system prompt del
   agente (`agent/src/fti_agent/system_prompt.py`) lo repite explícitamente.

3. **La restricción de un solo job de GPU se aplica server-side, no cliente-side.**
   `start_lora_training` llama a `training_service.is_training_active(conn)` y
   lanza `RuntimeError` si ya hay un job corriendo — incluyendo uno lanzado desde
   la UI de Streamlit por un humano. Un agente que ignora esta regla no corrompe
   nada: recibe el error y debe leerlo, no reintentar a ciegas.

**El MCP Host (`agent/`)** es un proyecto hermano independiente (propio
`pyproject.toml`/`.venv`/`uv.lock`) que consume ese servidor. Deliberadamente
**no usa LangGraph** — es un bucle escrito a mano (`agent/src/fti_agent/loop.py`):
modelo → `tool_calls` → ejecución → `ToolMessage` → repetir, con un límite duro de
iteraciones. `langchain-mcp-adapters` descubre y convierte las 11 tools MCP a objetos
`BaseTool` de LangChain, y envuelve automáticamente cualquier error de ejecución MCP
en `ToolMessage(status="error")` en vez de lanzar una excepción — el modelo ve un
fallo de tool como un turno normal de conversación que puede leer y ante el cual
puede reaccionar, no como un crash. La selección de proveedor de LLM es agnóstica vía
`langchain.chat_models.init_chat_model` (string `"anthropic:claude-sonnet-5"`,
`"openai:gpt-4.1"`, etc.), con soporte adicional para servidores OpenAI-compatible
locales (LM Studio, Ollama) vía `ChatOpenAI(base_url=...)`.

---

## 3. Fundamentos Matemáticos y Algorítmicos

### 3.1 Deduplicación perceptual: pHash, dHash y el Color Guard de 42 bits

El módulo de calidad no compara bytes de archivo — compara **percepción visual**
mediante hashes perceptuales de la librería `imagehash`, calculados una sola vez en
`image_service.compute_image_metrics` y almacenados en `ImageMetrics`.

**pHash (Perceptual Hash)** reduce la imagen a su firma de baja frecuencia en el
dominio de la frecuencia (DCT — Discrete Cosine Transform): conserva los coeficientes
de mayor energía (las formas y gradientes generales de la imagen) y descarta el
detalle de alta frecuencia (textura fina, ruido de compresión). El resultado es un
hash de 64 bits que es **robusto a redimensionado, recompresión JPEG y pequeños
ajustes de brillo/contraste** — exactamente las variaciones que hacen que dos copias
de la "misma" foto no sean bit-idénticas mientras sí son perceptualmente idénticas.

**dHash (Difference Hash)** trabaja distinto: reduce la imagen a una grilla pequeña
(típicamente 9×8 píxeles) y codifica un bit por cada comparación horizontal entre
píxeles adyacentes (`1` si el píxel de la derecha es más brillante, `0` si no). Es
más sensible a la estructura de gradientes locales que pHash, lo que lo hace un buen
**segundo factor**: dos imágenes que casualmente coinciden en pHash (falso positivo)
raramente coinciden también en dHash.

**La combinación en `quality_service.perceptual_distance`:**

```python
distance = hamming_distance(left.metrics.phash, right.metrics.phash)
if left.metrics.dhash and right.metrics.dhash:
    distance = max(distance, hamming_distance(left.metrics.dhash, right.metrics.dhash))
```

Tomar el **máximo** entre las dos distancias de Hamming (no el promedio, no el
mínimo) es la decisión matemática clave: un par de imágenes solo cuenta como
duplicado si **ambas** señales coinciden por debajo del umbral
(`DEFAULT_PHASH_THRESHOLD = 5`, sobre un espacio de distancias 0-64 para hashes de 64
bits). Si una señal dice "muy similar" (distancia baja) pero la otra dice "distinto"
(distancia alta), el máximo empuja la distancia combinada hacia arriba y el par no
clusteriza.

**El Color Guard de 42 bits** existe porque tanto pHash como dHash operan sobre
**luminancia** (escala de grises) — dos imágenes con la misma composición pero
colores completamente distintos (un render de estudio en dos paletas, un fondo liso
azul vs. verde) pueden hashear idéntico en pHash/dHash a pesar de ser visualmente
diferentes. El colorhash de `imagehash` codifica la distribución de tonos/saturación
como un flat-hash de 42 bits (decodificado con `imagehash.hex_to_flathash(h, 42)` —
distinto formato al de pHash/dHash, que son matrices cuadradas). El guard actúa como
**veto**, no como señal aditiva:

```python
if color_distance(left.colorhash, right.colorhash) > COLOR_GUARD_DISTANCE:  # 4
    return MAX_DISTANCE  # 64 — nunca clusterizan
```

Si el color diverge más allá del umbral, la función corta corto y fuerza la distancia
al máximo posible, sin siquiera calcular pHash/dHash. Esto es lo que evita que dos
fotos de estudio con fondos de color distinto —estructuralmente casi idénticas— se
marquen como duplicados.

El clustering en sí (`find_duplicate_clusters`) es un **pase único greedy O(n²)**: el
primer sample no clusterizado de la lista se propone como "keep", y todo sample
posterior a distancia ≤ umbral de él se une a su clúster y se marca visto. No es
clustering jerárquico ni transitividad completa — es intencional y documentado: es
la asignación práctica, no la teóricamente óptima, y el costo cuadrático es aceptable
porque corre sobre un run (cientos de imágenes), no sobre el dataset entero.

### 3.2 Nitidez: Varianza del Laplaciano

`image_service.compute_sharpness` mide qué tan enfocada está una imagen calculando
la **varianza del operador Laplaciano** — la medida clásica de detección de blur en
visión por computador.

El Laplaciano es un operador de derivada segunda que resalta bordes: en una región
plana y sin detalle, la segunda derivada es ~0 en todas partes; en un borde nítido,
tiene un pico pronunciado. La intuición es directa: **una imagen desenfocada tiene
bordes suaves → poca variación en la respuesta del Laplaciano → varianza baja. Una
imagen nítida tiene bordes marcados → alta variación → varianza alta.**

La implementación aplica el kernel discreto de 4 vecinos directamente sobre un array
de `float64`, sin pasar por `PIL.ImageFilter.Kernel`:

```python
laplacian = (
    pixels[:-2, 1:-1] + pixels[2:, 1:-1] +      # vecino arriba + abajo
    pixels[1:-1, :-2] + pixels[1:-1, 2:] -      # vecino izquierda + derecha
    4 * pixels[1:-1, 1:-1]                       # -4 × centro
)
return float(laplacian.var())
```

Esto es el kernel clásico `[[0,1,0],[1,-4,1],[0,1,0]]` aplicado por convolución
vectorizada con slicing de NumPy. La razón para no usar `ImageFilter.Kernel` de PIL
está documentada explícitamente: ese filtro **clampea a [0, 255]** sobre imágenes de
8 bits, lo que descartaría la mitad negativa de la respuesta del Laplaciano —
destruyendo precisamente la información de contraste que la varianza necesita medir.

Dos decisiones adicionales, ambas con trade-offs explícitos en el código:

- **La imagen se reduce a un thumbnail de 512px antes de medir** —
  `grey.thumbnail((512, 512), Image.Resampling.LANCZOS)`. La varianza del Laplaciano
  escala con la resolución (más píxeles → más oportunidades de bordes fuertes), así
  que sin normalizar el tamaño, una foto grande y suave podría superar en "nitidez"
  a una pequeña y nítida — números no comparables dentro del mismo set.

- **El resultado solo es comparable *dentro de un mismo run/concept*, nunca entre
  datasets.** El propio docstring lo advierte: *"texture and subject matter move it
  as much as focus does, so there is no threshold that means 'blurry' across
  datasets."* Por eso `quality_service.blurriest_samples` es deliberadamente un
  **ranking** ("las 6 más suaves de este set"), no un umbral fijo tipo
  `sharpness < X ⇒ borrosa`.

### 3.3 LoRA: descomposición de bajo rango en el fine-tuning (Paso 5)

El entrenamiento (`workers/train_worker.py`) usa **LoRA (Low-Rank Adaptation)** vía
`peft.LoraConfig` sobre el transformer de difusión (Krea 2), no fine-tuning completo.

**La idea matemática.** Fine-tuning completo actualiza cada matriz de pesos
`W ∈ ℝ^(d×k)` de una capa lineal directamente: `W' = W + ΔW`, donde `ΔW` tiene tantos
parámetros como `W` misma (`d × k`). Para un transformer de miles de millones de
parámetros, eso es prohibitivamente caro de entrenar y de almacenar por cada
concepto/estilo.

LoRA parte de la observación empírica (Hu et al., 2021) de que la actualización
`ΔW` necesaria para adaptar un modelo grande a una tarea nueva tiene **rango
intrínseco bajo** — no necesita explorar el espacio completo de `d × k` grados de
libertad. En vez de aprender `ΔW` directamente, LoRA la **factoriza** como el
producto de dos matrices mucho más pequeñas:

```
ΔW = B · A         donde A ∈ ℝ^(r×k),  B ∈ ℝ^(d×r),   r ≪ min(d, k)
```

`A` se inicializa con ruido gaussiano pequeño, `B` se inicializa en **cero** — así,
al arrancar el entrenamiento, `ΔW = B·A = 0` y el modelo se comporta exactamente
como el checkpoint base sin adaptar. Durante el fine-tuning, `W` original queda
**congelado** (no recibe gradiente) y solo `A` y `B` se entrenan. El forward pass de
la capa adaptada es:

```
y = W·x + (α/r) · B·A·x
```

El término `(α/r)` es el **factor de escala**: `α` (alpha) es un hiperparámetro fijo
y `r` es el rango elegido, de forma que el impacto de la actualización LoRA se
normaliza independientemente de qué rango se use — sin este escalado, subir `r`
cambiaría implícitamente cuánto "empuja" la adaptación, acoplando dos decisiones que
deberían ser independientes.

**En el código:** `train_worker.py` construye la config con
`LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=...,
use_dora=False, init_lora_weights=True)` y la aplica al transformer con
`get_peft_model(transformer, lora_config)`. `target_modules` se calcula filtrando las
capas lineales del transformer por nombre (`LORA_TARGET`), no se adapta el modelo
entero — típicamente las proyecciones de atención (`to_q`, `to_k`, `to_v`, `to_out`)
y/o las capas del bloque feed-forward, que es donde la literatura de LoRA reporta el
mejor costo/beneficio.

**Impacto práctico de los hiperparámetros, tal como los expone el pipeline**
(`TrainingConfig` en `training_service.py`, defaults `lora_rank=16`,
`lora_alpha=32`):

- **Rango (`r`)** controla la **capacidad expresiva** de la adaptación. Un `r` bajo
  (4-8) fuerza a `ΔW` a vivir en un subespacio muy comprimido — el LoRA aprende
  menos matices pero es rápido de entrenar, pequeño en disco (unos pocos MB), y
  menos propenso a sobreajustar con datasets pequeños (que es exactamente el
  escenario de este pipeline: decenas a cientos de imágenes por concepto). Un `r`
  alto (32-64+) permite capturar transformaciones más complejas del estilo/sujeto,
  a costa de más parámetros entrenables, más VRAM, y más riesgo de overfitting si
  el dataset curado es pequeño o poco diverso — precisamente el motivo por el que
  el pipeline invierte tanto en el paso 3 (Quality): un dataset con duplicados o
  captions vacíos amplifica ese riesgo.

- **Alpha (`α`)**, vía el factor `α/r`, controla la **magnitud del efecto** de la
  adaptación sobre las activaciones de la capa, independiente del rango elegido.
  El código expone directamente esta relación en su propio log de arranque
  (`train_worker.py:431`):

  ```python
  ("lora_rank", f"{LORA_RANK} / alpha {LORA_ALPHA} (scale {LORA_ALPHA / max(1, LORA_RANK):.2f})")
  ```

  Con los defaults (`r=16, α=32`), el factor de escala es `32/16 = 2.0`. Subir
  `alpha` sin subir `r` intensifica cuánto "tira" el LoRA sobre el modelo base con
  la misma capacidad expresiva — útil para conceptos donde el estilo debe dominar
  fuerte sobre el prior del modelo base, a riesgo de degradar la coherencia general
  si se pasa de rosca. La regla práctica más común en la comunidad (y la que los
  defaults de este pipeline siguen: `alpha = 2×rank`) es una escala de partida
  razonable, no una ley matemática — es terreno de experimentación, y por eso el
  pipeline expone ambos como hiperparámetros libres en vez de acoplarlos.

- **`lora_dropout=0.0`** y **`use_dora=False`**: el worker deliberadamente no usa
  dropout sobre las matrices LoRA ni la variante DoRA (Weight-Decomposed LoRA, que
  además de `B·A` aprende una magnitud por columna). Ambas son simplificaciones
  conscientes del scope — el vendored `train_worker.py` prioriza estabilidad y
  paridad con el LoRAlab original sobre exprimir cada técnica reciente.

**La trampa operacional que conecta esta matemática con la capa de orquestación
(§1.3, §2.3):** `continue_lora_training` **no recibe los hiperparámetros de la
llamada anterior** — cada llamada MCP es una construcción nueva de `TrainingConfig`
con sus propios defaults. Si `start_lora_training` se llamó con `lora_rank=32,
lora_alpha=64` pero `continue_lora_training` se llama sin especificar esos
parámetros, el train worker arrancará con los defaults (`16`, `32`) — un
`LoraConfig` con rango y escala completamente distintos a los que el precache
preparó, y **sin ningún error explícito**: simplemente entrena con una adaptación
de menor capacidad y menor magnitud de la que el operador pretendía. Es la razón
por la que el system prompt del agente MCP Host trata este acoplamiento como el
riesgo de mayor prioridad de toda la superficie de automatización.

---

## Cómo interactúan los tres pilares

Un ejemplo end-to-end que toca los tres:

1. **Import** calcula `ImageMetrics` (matemática: pHash/dHash/colorhash/Laplaciano)
   para cada imagen, usando `imagehash` + NumPy (tecnología), y las persiste como
   `DatasetSample` Pydantic validados (arquitectura: domain puro, cero I/O en el
   modelo mismo).

2. **Quality** compara esos metrics con la fórmula de distancia combinada + Color
   Guard (matemática) en funciones puras de `application/quality_service.py`
   (arquitectura), que la UI invoca vía `ui/state.py` y que el MCP server invoca vía
   `mcp_server.server.quality_summary` (tecnología/protocolo) — **la misma función**,
   dos consumidores.

3. **Export** hashea el contenido curado (matemática: SHA-256 sobre pares
   `(phash, caption)` ordenados) para producir un `DatasetManifest` (arquitectura:
   domain model) que vive en SQLite (tecnología).

4. **Train** lee ese dataset exportado, lo pasa a un `LoraConfig` con `r` y `α`
   elegidos por el operador (matemática: descomposición de bajo rango) dentro de un
   subprocess detached en otro intérprete (arquitectura: frontera de proceso,
   tecnología: PEFT/Accelerate), orquestable de extremo a extremo por un agente vía
   MCP (protocolo) que debe entender, sin que nadie se lo diga dos veces, que los
   hiperparámetros de esa fórmula matemática no viajan solos entre dos tool calls.

Esa última frase es, en cierto sentido, el resumen de por qué existen `agent/` y su
`system_prompt.py`: la arquitectura del sistema es sólida y las matemáticas son
correctas, pero el contrato entre ambas —qué información persiste entre llamadas y
qué no— es exactamente el tipo de detalle que un docstring de tool puede mencionar
pero que solo una capa de orquestación consciente del dominio completo puede hacer
cumplir de forma fiable.

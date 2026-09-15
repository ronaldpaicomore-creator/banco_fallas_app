import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

from drive_indexer import process_drive_folder, process_drive_folder_incremental
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma


def es_error_de_cuota(e: Exception) -> bool:
    texto = str(e)
    return "RESOURCE_EXHAUSTED" in texto or "429" in texto or "quota" in texto.lower()

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — lee de variables de entorno / .env, NUNCA hardcodees la API key
# ---------------------------------------------------------------------------
DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PERSIST_DIR = "chroma_db"
CHUNKS_CACHE = os.path.join(PERSIST_DIR, "chunks_cache.pkl")
PROGRESO_FILE = os.path.join(PERSIST_DIR, "progreso.txt")

st.set_page_config(page_title="Buscador de Fallas", layout="wide")
st.title("🛠️ Buscador de Documentos - Banco de Fallas")


@st.cache_resource(show_spinner=False)
def load_vectorstore(force_reindex: bool = False):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY
    )

    marcador = os.path.join(PERSIST_DIR, ".indexado_ok")

    if os.path.exists(marcador) and not force_reindex:
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    if force_reindex and os.path.isdir(PERSIST_DIR):
        import shutil
        import gc
        import time as _time

        # En Windows, si el índice anterior sigue "vivo" en memoria (de esta
        # misma app), el sistema operativo bloquea sus archivos y no deja
        # borrarlos de inmediato. Forzamos la liberación y reintentamos.
        gc.collect()
        for intento in range(5):
            try:
                shutil.rmtree(PERSIST_DIR)
                break
            except PermissionError:
                _time.sleep(2)
        else:
            st.error(
                "No se pudo reemplazar el índice anterior porque Windows lo tiene "
                "bloqueado. Cierra la app por completo (Ctrl+C en la terminal), "
                "vuelve a correr 'python -m streamlit run app.py', y recién ahí "
                "dale clic a 'Reindexar Drive' otra vez."
            )
            st.stop()

    os.makedirs(PERSIST_DIR, exist_ok=True)

    with st.spinner("Preparando documentos..."):
        # Si ya habíamos extraído y dividido los documentos en un intento
        # anterior (aunque se haya cortado a medias por una cuota agotada),
        # reutilizamos ese trabajo en vez de volver a leer todo el Drive.
        if os.path.exists(CHUNKS_CACHE):
            with open(CHUNKS_CACHE, "rb") as f:
                chunks = pickle.load(f)
        else:
            raw_documents = process_drive_folder(DRIVE_FOLDER_ID)

            if not raw_documents:
                st.error(
                    "No se encontraron documentos legibles en la carpeta. "
                    "Revisa que la cuenta de servicio tenga acceso y que haya "
                    "PDFs, Word o PowerPoint dentro de las subcarpetas."
                )
                st.stop()

            # Chunks más grandes = menos llamadas totales a la API de
            # embeddings, lo cual ayuda a no toparse con la cuota gratuita.
            splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
            chunks = splitter.split_documents(raw_documents)

            with open(CHUNKS_CACHE, "wb") as f:
                pickle.dump(chunks, f)

    # ¿Ya hay progreso guardado de un intento anterior? Retomamos desde ahí
    # en vez de volver a gastar cuota embebiendo lo que ya se guardó.
    inicio = 0
    if os.path.exists(PROGRESO_FILE):
        with open(PROGRESO_FILE) as f:
            inicio = int(f.read().strip() or 0)

    BATCH_SIZE = 15
    PAUSE_SECONDS = 20

    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    total = len(chunks)
    progreso = st.progress(inicio / total if total else 0, text=f"Generando embeddings: {inicio}/{total}")

    for i in range(inicio, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        intentos = 0
        while True:
            try:
                vectordb.add_documents(batch)
                break
            except Exception as e:
                if not es_error_de_cuota(e):
                    raise
                intentos += 1
                if intentos >= 3:
                    # Guardamos dónde nos quedamos para poder retomar
                    # exactamente aquí la próxima vez, sin perder lo ya hecho.
                    with open(PROGRESO_FILE, "w") as f:
                        f.write(str(i))
                    vectordb.persist()
                    st.error(
                        "Se agotó la cuota gratuita de la API de Gemini por hoy. "
                        f"Ya se indexaron {i} de {total} fragmentos y quedaron guardados — "
                        "cuando corras la app de nuevo (hoy más tarde o mañana), "
                        "va a continuar justo desde aquí, no desde cero."
                    )
                    st.stop()
                espera = 30 * intentos
                progreso.progress(
                    min(i / total, 1.0),
                    text=f"Cuota momentánea alcanzada, esperando {espera}s antes de reintentar...",
                )
                time.sleep(espera)

        hecho = min(i + BATCH_SIZE, total)
        progreso.progress(hecho / total, text=f"Generando embeddings: {hecho}/{total}")
        # Guardamos el avance por si el proceso se interrumpe de golpe
        with open(PROGRESO_FILE, "w") as f:
            f.write(str(hecho))
        time.sleep(PAUSE_SECONDS)

    vectordb.persist()
    with open(marcador, "w") as f:
        f.write("ok")
    # Limpieza: ya no necesitamos los archivos temporales de progreso
    for tmp in (CHUNKS_CACHE, PROGRESO_FILE):
        if os.path.exists(tmp):
            os.remove(tmp)
    progreso.empty()
    return vectordb


def score_to_percent(score: float) -> int:
    # Chroma devuelve "distancia" (menor = más parecido). Lo convertimos
    # a un porcentaje aproximado solo para mostrar al técnico.
    pct = max(0, min(100, round((1 - score / 2) * 100)))
    return pct


# Palabras clave para reconocer una ESPECIALIDAD real dentro de la ruta de
# carpetas. Todo lo que aparezca en la ruta o el nombre del archivo y NO
# coincida con esta lista se trata como candidato a "máquina" en su lugar.
# Si te falta o te sobra alguna, avísame y la ajustamos.
ESPECIALIDADES_CONOCIDAS = [
    "Eléctrica", "Electrica", "Mecánica", "Mecanica", "Soplado",
    "Automatización", "Automatizacion", "Instrumentación", "Instrumentacion",
    "Neumática", "Neumatica", "Electrónica", "Electronica",
]

# Nombres de máquina a reconocer dentro de la ruta de carpetas o del
# nombre del archivo. Agrega aquí cualquier máquina que falte.
MAQUINAS_CONOCIDAS = [
    "Llenadora", "Sopladora", "Etiquetadora", "Empacadora", "Envolvedora",
    "Paletizadora", "Mezcladora", "Mixer", "Ionizador", "Inspector de nivel",
    "Codificadora", "Transportador", "Compresor", "Enfardadora",
    "Formadora", "Selladora", "Rotuladora", "Fuerza", "Datamatrix",
    "Encajonadora", "Desencajonadora", "Rinser", "Taponadora",
]


def _normalizar(texto: str) -> str:
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )


def detectar_etiquetas(ruta: str, nombre: str):
    """Busca en la ruta + nombre del archivo qué especialidad y qué
    máquina(s) conocidas aparecen mencionadas."""
    texto = _normalizar(f"{ruta} {nombre}")

    especialidad = next(
        (e for e in ESPECIALIDADES_CONOCIDAS if _normalizar(e) in texto), None
    )
    maquinas = [m for m in MAQUINAS_CONOCIDAS if _normalizar(m) in texto]

    return especialidad, maquinas


@st.cache_data(show_spinner=False)
def obtener_valores_filtros(_vectordb):
    """Lee toda la metadata guardada para armar las listas de filtros:
    línea (de carpeta), especialidad real y máquina (detectadas por
    palabra clave en ruta + nombre de archivo)."""
    datos = _vectordb._collection.get(include=["metadatas"])
    lineas = sorted({m.get("linea", "") for m in datos["metadatas"] if m.get("linea")})

    especialidades = set()
    maquinas = set()
    for m in datos["metadatas"]:
        esp, maqs = detectar_etiquetas(m.get("ruta", ""), m.get("nombre", ""))
        if esp:
            especialidades.add(esp)
        maquinas.update(maqs)

    return lineas, sorted(especialidades), sorted(maquinas)


def generar_diagnostico_stream(pregunta: str, contexto_docs):
    """Genera el diagnóstico palabra por palabra (streaming) en vez de
    esperar la respuesta completa, para que se sienta más rápido."""
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=GEMINI_API_KEY)
    contexto = "\n\n---\n\n".join(
        f"Documento: {d.metadata['nombre']}\nContenido: {d.page_content[:1500]}"
        for d, _ in contexto_docs
    )
    prompt = f"""Eres un asistente técnico de mantenimiento industrial. Un técnico busca
resolver esta falla: "{pregunta}"

Con base ÚNICAMENTE en los siguientes fragmentos extraídos del Banco de Fallas,
da una lista corta y numerada (máx. 4 pasos) de la intervención recomendada.
Sé concreto y técnico. Si el contexto no alcanza, dilo.

Contexto:
{contexto}
"""
    for chunk in llm.stream(prompt):
        contenido = chunk.content
        if isinstance(contenido, list):
            contenido = "".join(
                bloque.get("text", "") for bloque in contenido if isinstance(bloque, dict)
            )
        if contenido:
            yield contenido


def _embeber_por_lotes(vectordb, chunks, progreso_widget=None):
    """Agrega chunks a un vectordb ya existente, en lotes pequeños con
    pausas y reintentos ante cuota agotada. Reutilizado tanto por el
    indexado completo como por el incremental."""
    BATCH_SIZE = 15
    PAUSE_SECONDS = 20
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        intentos = 0
        while True:
            try:
                vectordb.add_documents(batch)
                break
            except Exception as e:
                if not es_error_de_cuota(e):
                    raise
                intentos += 1
                if intentos >= 3:
                    vectordb.persist()
                    st.error(
                        "Se agotó la cuota gratuita de la API de Gemini. "
                        f"Se agregaron {i} de {total} fragmentos nuevos antes de cortarse. "
                        "Vuelve a darle clic a 'Agregar archivos nuevos' más tarde para "
                        "completar el resto (los archivos que ya se agregaron no se repiten)."
                    )
                    st.stop()
                espera = 30 * intentos
                if progreso_widget:
                    progreso_widget.progress(
                        min(i / total, 1.0), text=f"Cuota alcanzada, esperando {espera}s..."
                    )
                time.sleep(espera)

        if progreso_widget:
            hecho = min(i + BATCH_SIZE, total)
            progreso_widget.progress(hecho / total, text=f"Generando embeddings: {hecho}/{total}")
        time.sleep(PAUSE_SECONDS)

    vectordb.persist()


def indexar_archivos_nuevos(vectordb):
    """Revisa el Drive y solo agrega los archivos que todavía no están en
    el índice, sin tocar (ni volver a gastar cuota en) los que ya existen."""
    datos = vectordb._collection.get(include=["metadatas"])
    ids_ya_indexados = {m["id"] for m in datos["metadatas"] if m.get("id")}
    # Respaldo para archivos indexados antes de que existiera el campo "id"
    pares_ya_indexados = {
        (m.get("nombre", ""), m.get("ruta", "")) for m in datos["metadatas"]
    }

    with st.spinner("Buscando archivos nuevos en Drive..."):
        nuevos_docs = process_drive_folder_incremental(
            DRIVE_FOLDER_ID, ids_ya_indexados, pares_ya_indexados
        )

    if not nuevos_docs:
        st.success("No hay archivos nuevos por agregar — tu índice ya está al día.")
        return

    st.info(f"Se encontraron {len(nuevos_docs)} archivo(s) nuevo(s). Generando embeddings...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_documents(nuevos_docs)

    progreso = st.progress(0, text=f"Generando embeddings: 0/{len(chunks)}")
    _embeber_por_lotes(vectordb, chunks, progreso)
    progreso.empty()
    st.success(f"Listo — se agregaron {len(nuevos_docs)} archivo(s) nuevo(s) al índice.")


# ---------------------------------------------------------------------------
# Barra lateral: reindexar manualmente cuando se agreguen archivos nuevos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Administración")
    if st.button("🔄 Reindexar todo (desde cero)"):
        st.cache_resource.clear()
        load_vectorstore(force_reindex=True)
        st.success("Reindexado completo.")

vectordb = load_vectorstore()

with st.sidebar:
    if st.button("➕ Agregar archivos nuevos"):
        indexar_archivos_nuevos(vectordb)
        st.cache_data.clear()  # refresca las listas de filtros con los datos nuevos

with st.sidebar:
    try:
        total_en_indice = vectordb._collection.count()
        st.caption(f"📦 Fragmentos en el índice: {total_en_indice}")
    except Exception as e:
        st.caption(f"No se pudo leer el índice: {e}")

# ---------------------------------------------------------------------------
# Dashboard informativo: cuántos ARCHIVOS únicos hay en total y por especialidad
# (un archivo puede tener varios fragmentos, así que contamos nombres únicos)
# ---------------------------------------------------------------------------
with st.expander("📊 Resumen del Banco de Fallas", expanded=False):
    datos_todos = vectordb._collection.get(include=["metadatas"])
    archivos_vistos = {}
    for m in datos_todos["metadatas"]:
        clave = (m.get("nombre", ""), m.get("ruta", ""))
        if clave in archivos_vistos:
            continue
        esp, _ = detectar_etiquetas(m.get("ruta", ""), m.get("nombre", ""))
        archivos_vistos[clave] = esp or "Sin especialidad detectada"

    total_archivos = len(archivos_vistos)
    st.metric("Total de archivos", total_archivos)

    conteo_por_especialidad = {}
    for esp in archivos_vistos.values():
        conteo_por_especialidad[esp] = conteo_por_especialidad.get(esp, 0) + 1

    cols_dash = st.columns(len(conteo_por_especialidad) or 1)
    for col, (esp, cant) in zip(cols_dash, sorted(conteo_por_especialidad.items())):
        col.metric(esp, cant)

# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
lineas_disponibles, especialidades_disponibles, maquinas_disponibles = obtener_valores_filtros(vectordb)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    filtro_linea = st.selectbox("Línea", ["Todas"] + lineas_disponibles)
with col_f2:
    filtro_especialidad = st.selectbox("Especialidad", ["Todas"] + especialidades_disponibles)
with col_f3:
    filtro_maquina = st.selectbox("Máquina", ["Todas"] + maquinas_disponibles)

query = st.text_input("Describe la falla", placeholder="Ej: Falla comunicación válvulas electrónicas llenadora L1")

buscar = st.button("🔍 BUSCAR", type="primary")

if buscar and query.strip():
    # La línea sí está guardada como campo exacto en el índice, así que
    # filtramos por ahí directo.
    filtro_chroma = {"linea": filtro_linea} if filtro_linea != "Todas" else None

    # Especialidad y máquina se detectan por palabra clave (no son un campo
    # fijo del índice), así que traemos más candidatos de los necesarios
    # y filtramos en Python antes de quedarnos con el top 5.
    candidatos = vectordb.similarity_search_with_score(query, k=30, filter=filtro_chroma)

    resultados = []
    for doc, score in candidatos:
        esp, maqs = detectar_etiquetas(doc.metadata.get("ruta", ""), doc.metadata.get("nombre", ""))
        if filtro_especialidad != "Todas" and esp != filtro_especialidad:
            continue
        if filtro_maquina != "Todas" and filtro_maquina not in maqs:
            continue
        resultados.append((doc, score))
        if len(resultados) == 5:
            break

    if not resultados:
        st.warning(
            "No se encontraron coincidencias con esos filtros. "
            "Prueba con 'Todas' en Línea/Especialidad/Máquina o reformula la búsqueda."
        )
        st.stop()

    col_ia, col_docs = st.columns([1, 1.3])

    # Mostramos primero las coincidencias de Drive — esto es instantáneo
    # porque la búsqueda ocurre en tu propia máquina, no depende de la IA.
    with col_docs:
        st.subheader("📄 Coincidencias en Banco de Fallas (Google Drive)")
        for doc, score in resultados:
            meta = doc.metadata
            pct = score_to_percent(score)
            with st.container(border=True):
                st.markdown(f"**{meta['nombre']}**  `{pct}% coincidencia`")
                st.caption(f"Ubicación: {meta['ruta']}")
                snippet = doc.page_content[:220].replace("\n", " ")
                st.markdown(f"> _{snippet}..._")
                if meta.get("link"):
                    st.link_button("🔗 Abrir en Google Drive", meta["link"])

    # El diagnóstico de la IA va al final porque es lo que más tarda
    # (necesita ir y volver a los servidores de Google) — así el técnico
    # ya está viendo y puede abrir los documentos mientras esto termina.
    with col_ia:
        st.subheader("🤖 Diagnóstico y Solución Rápida (IA)")
        st.write_stream(generar_diagnostico_stream(query, resultados[:3]))
        fuente = resultados[0][0].metadata
        st.info(f"📌 Documento fuente original: **{fuente['nombre']}** ({fuente['ruta']})")
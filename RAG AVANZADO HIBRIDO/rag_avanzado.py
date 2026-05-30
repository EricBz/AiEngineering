import os
import json
import warnings
from dotenv import load_dotenv

# Ocultamos avisos de inicialización y logs internos para limpiar la consola
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. CARGA SEGURA DE CREDENCIALES
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ ERROR: No se encontró la variable GEMINI_API_KEY en el archivo .env")

# 2. IMPORTACIONES ROBUSTAS Y 100% ACTUALIZADAS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Ruta oficial moderna
from langchain_core.documents import Document
from google import genai
from google.genai import types

# Inicializamos el cliente oficial de Google GenAI
client = genai.Client()

# 3. CONJUNTO DE DATOS DE PRUEBA
documentos_crudos = [
    "El sistema falló debido a un problema de conexión en la base de datos.",
    "Manual técnico: El código de error ERR-404 indica que el usuario no fue encontrado.",
    "Guía de red: Los fallos de timeout ocurren cuando el servidor no responde a tiempo.",
    "El código ERR-404 también se genera si la base de datos está caída durante el login."
]
docs = [Document(page_content=t, metadata={"id": i}) for i, t in enumerate(documentos_crudos)]

query = "Necesito soporte para el error ERR-404 en la base de datos"
print(f"🔍 Consulta planteada: '{query}'\n")

# =====================================================================
# ETAPA 1: BÚSQUEDA HÍBRIDA (Chroma Local + Algoritmo BM25 Nativo)
# =====================================================================
print("⚡ Ejecutando Búsqueda Híbrida...")

# Búsqueda Vectorial Local Moderna (Sin advertencias)
embeddings_locales = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings_locales)
resultados_vectoriales = vectorstore.similarity_search(query, k=2)

# Búsqueda Léxica por Palabras Clave Exactas (BM25)
from rank_bm25 import BM25Okapi
corpus_tokenizado = [doc.page_content.lower().split(" ") for doc in docs]
bm25 = BM25Okapi(corpus_tokenizado)
query_tokenizada = query.lower().split(" ")
mejores_docs_bm25 = bm25.get_top_n(query_tokenizada, documentos_crudos, n=2)

# Combinación de Candidatos sin duplicados
candidatos_unicos = set()
for doc in resultados_vectoriales:
    candidatos_unicos.add(doc.page_content)
for doc_texto in mejores_docs_bm25:
    candidatos_unicos.add(doc_texto)

lista_candidatos = list(candidatos_unicos)

print("\n📌 [Fase Intermedia] Candidatos recuperados de forma híbrida:")
for i, texto in enumerate(lista_candidatos):
    print(f"   {i+1}. {texto}")
print("-" * 60)

# =====================================================================
# ETAPA 2: RE-RANKING SEMÁNTICO (Gemini 2.5 Flash como Cross-Encoder)
# =====================================================================
print("🧠 Ejecutando Re-ranking de precisión con Gemini...")

prompt_reranker = f"""
Actúa como un sistema experto de re-ranking de información (Cross-Encoder).
Evalúa la relevancia semántica de cada documento candidato para resolver la consulta del usuario.

Consulta: "{query}"

Documentos a evaluar:
{chr(10).join([f"[{i}] {texto}" for i, texto in enumerate(lista_candidatos)])}

Devuelve única y exclusivamente un objeto JSON con la clave "ranking" que contenga la lista de índices ordenados de mayor a menor importancia. No agregues código markdown ni explicaciones adicionales.
"""

# Invocación estructurada nativa a la API de Google GenAI
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt_reranker,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0
    ),
)

# Mapeo y ordenamiento final
resultado_json = json.loads(response.text)
indices_ordenados = resultado_json["ranking"]

print("\n🏆 [Fase Final] Resultados reordenados con éxito:")
for lugar, idx in enumerate(indices_ordenados):
    print(f"   Puesto {lugar + 1}: '{lista_candidatos[idx]}'")





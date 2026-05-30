import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

#Para instalar pip install google-genai chromadb


# 1. Configura tu API Key de Gemini
# El SDK de Chroma busca automáticamente esta variable de entorno
os.environ["GEMINI_API_KEY"] = "tu-api-key-de-gemini-aqui"

# 2. Inicializar el cliente de ChromaDB local
client = chromadb.PersistentClient(path="./mi_base_vectorial_gemini")

# 3. Definir la función de embedding utilizando el modelo de Google
# Usamos 'gemini-embedding-001' optimizado para tareas de recuperación de documentos
funcion_gemini = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT"
)

# 4. Crear la colección vinculándole la función de Google
coleccion = client.get_or_create_collection(
    name="documentos_gemini",
    embedding_function=funcion_gemini
)

# 5. Agregar documentos
# Chroma enviará automáticamente estos textos a los servidores de Google
coleccion.add(
    documents=[
        "El aprendizaje profundo es un subcampo de la inteligencia artificial.",
        "Las manzanas, plátanos y naranjas son frutas deliciosas.",
        "Los algoritmos de búsqueda vectorial encuentran datos por su significado semántico."
    ],
    ids=["id1", "id2", "id3"]
)

# 6. Consultar
# La pregunta también viaja a Google para convertirse en vector antes de buscar
resultado = coleccion.query(
    query_texts=["Háblame sobre redes neuronales y machine learning"],
    n_results=1
)

print("Documento más cercano encontrado mediante Gemini:")
print(resultado["documents"])

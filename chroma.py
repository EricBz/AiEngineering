import chromadb

# 1. Inicializar el cliente (se guarda en memoria por defecto)
cliente = chromadb.Client()

# 2. Crear una colección
coleccion = cliente.create_collection(name="documentos_semanticos")

# 3. Agregar vectores y textos (Chroma genera los embeddings automáticamente)
coleccion.add(
    documents=[
        "El clima en Buenos Aires es templado.",
        "Python es un lenguaje de programación.",
        "Las bases de datos vectoriales buscan similitudes."
    ],
    metadatas=[{"categoria": "clima"}, {"categoria": "programacion"}, {"categoria": "IA"}],
    ids=["doc1", "doc2", "doc3"]
)

# 4. Realizar una búsqueda de similitud semántica
resultados = coleccion.query(
    query_texts=["¿Para qué sirve una base de datos vectorial?"],
    n_results=1
)

print(resultados)

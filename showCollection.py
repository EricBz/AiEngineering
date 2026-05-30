import chromadb

# 1. Inicializa el cliente apuntando al directorio de tu base de datos
client = chromadb.PersistentClient(path="./BBVECTORES")
#Creacion de la coleccion
coleccion = client.create_collection(name="documentos_semanticos")
#Se agregan documentos a la coleccion.
coleccion.add(
    documents=[
        "El clima en Buenos Aires es templado.",
        "Python es un lenguaje de programación.",
        "Las bases de datos vectoriales buscan similitudes."
    ],
    metadatas=[{"categoria": "clima"}, {"categoria": "programacion"}, {"categoria": "IA"}],
    ids=["doc1", "doc2", "doc3"]
)
# 2. Accede a tu colección existente
coleccion = client.get_collection(name="documentos_semanticos")

# 3. (Opcional) Verifica los datos que contiene
print(coleccion.peek()) # Muestra los primeros 10 elementos
print(coleccion.count()) # Cuenta el total de documentos

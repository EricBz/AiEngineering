# Laboratorio Avanzado: Arquitectura RAG de Nueva Generación
## Búsqueda Híbrida y Re-ranking Semántico con Gemini 2.5 Flash

### 🎯 Objetivo del Ejercicio
Aprender a mitigar las limitaciones de la búsqueda vectorial tradicional mediante la implementación de un sistema de recuperación en dos etapas (*Two-Stage Retrieval*). Al finalizar, entenderás cómo combinar el poder conceptual de los vectores con la precisión de las palabras clave, optimizando la lista final de candidatos mediante un modelo de lenguaje avanzado actuando como *Cross-Encoder*.

---

### 💡 El Problema de la Búsqueda Tradicional
Los sistemas RAG convencionales confían ciegamente en los *embeddings* vectoriales. Sin embargo, los vectores tienden a fallar cuando el usuario realiza consultas que mezclan intenciones conceptuales con identificadores exactos (como códigos de error, números de serie o IDs de productos).

* **Consulta del usuario**: *"Necesito soporte para el error ERR-404 en la base de datos"*
* **Falla del Vector Puro**: Identifica que el usuario habla de "bases de datos", pero diluye la importancia del código específico `ERR-404`, trayendo documentos genéricos sobre caídas de sistemas.

---

### 🏗️ Arquitectura del Sistema del Ejercicio

El ejercicio simula un flujo de trabajo industrial dividido en dos fases críticas:
[ Consulta del Usuario ]│┌──────────────┴──────────────┐▼                             ▼【 Búsqueda Vectorial 】         【 Búsqueda Léxica 】(Chroma Local)                (Algoritmo BM25)Entiende el contexto          Busca códigos exactos│                             │└──────────────┬──────────────┘▼[ Fusión de Candidatos Únicos ]│▼【 Re-ranking con Gemini 】(Modelo Cross-Encoder)Ordena por relevancia quirúrgica│▼[ Resultados Optimizados ]
---

### 🔍 Desglose del Paso a Paso del Código

#### 1. Inicialización y Seguridad (`dotenv`)
Utilizamos `python-dotenv` para mantener las credenciales de infraestructura fuera del código fuente. Se configuran filtros de advertencias para garantizar una salida de consola limpia y profesional en entornos de producción.

#### 2. Etapa 1: Búsqueda Híbrida (Extracción de Candidatos)
Para no saturar la ventana de contexto del LLM ni desperdiciar tokens, lanzamos una "red de pesca" amplia utilizando dos motores en paralelo:
* **Motor Semántico**: `Chroma` procesa los textos de forma local utilizando el modelo de embeddings `all-MiniLM-L6-v2` para identificar la temática general.
* **Motor Léxico**: El algoritmo matemático nativo `BM25Okapi` escanea el texto buscando coincidencias exactas de caracteres para asegurar que el término `ERR-404` no se pierda.
* **Fusión**: Se combinan los resultados eliminando duplicados mediante conjuntos (`set`), generando una lista compacta de documentos preseleccionados.

#### 3. Etapa 2: Re-ranking Semántico (Filtrado Quirúrgico)
La lista híbrida intermedia puede contener documentos relevantes pero desordenados. Aquí entra **Gemini 2.5 Flash** actuando como un evaluador de atención cruzada (*Cross-Encoder*):
* Se le envía la consulta original junto con la lista de candidatos preseleccionados.
* Mediante un prompt técnico estricto y la configuración nativa `response_mime_type="application/json"`, se fuerza al modelo a devolver una matriz ordenada de índices basada puramente en lógica de relevancia.
* **Resultado Final**: El documento que integra tanto el concepto de la base de datos como el código alfanumérico exacto escala automáticamente al **Puesto 1**, listo para alimentar la fase de generación del RAG.




🛠️ Guía de Ejecución Rápida (Para el Docente)Sigue estos 4 pasos en tu computadora para poner a correr el laboratorio final optimizado:Preparar el directorio: Crea una carpeta nueva llamada clase_rag en tu máquina y ábrela en tu editor de código (como VS Code).Configurar las credenciales: Crea un archivo llamado exactamente .env dentro de esa carpeta y añade tu clave de API sin comillas ni espacios:envGEMINI_API_KEY=AIzaSyTuClaveRealDeGeminiAqui
Usa el código con precaución.Instalar el entorno limpio: Abre la terminal de tu editor y ejecuta el siguiente comando para limpiar librerías obsoletas e instalar los paquetes modernos oficiales:bashpip uninstall -y langchain-community langchain-google-genai
pip install python-dotenv langchain langchain-core langchain-chroma langchain-huggingface google-genai rank_bm25 sentence-transformers
Usa el código con precaución.Ejecutar el script: Guarda el último código corregido en un archivo llamado rag_avanzado.py y ejecútalo con python rag_avanzado.py. Verás la salida limpia y ordenada en segundos.

#choma funciona en memoria ram sin persistencia.


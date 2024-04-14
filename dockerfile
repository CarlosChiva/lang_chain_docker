FROM python:3.9

# Crear el directorio de trabajo
RUN mkdir /app
WORKDIR /app

# Instalar las dependencias
RUN pip install torch
RUN pip install transformers accelerate optimum
RUN pip install gpt4all
RUN pip install arxiv
RUN pip install pymupdf
RUN pip install langchain-cli
RUN pip install qdrant-client
RUN pip install pypdf unstructured
RUN pip install arxiv
RUN pip install ipykernel jupyter

# Crear la nueva aplicación LangChain
RUN echo "y" | langchain app new my-app --package rag-chroma-private

# Copiar los archivos personalizados
COPY server.py /app/my-app/app/server.py
COPY chain.py /app/my-app/packages/rag-chroma-private/rag_chroma_private/chain.py

# Ejecutar la aplicación LangChain
WORKDIR /app/my-app
CMD langchain serve
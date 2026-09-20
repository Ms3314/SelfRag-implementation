from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from self_rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PERSIST_DIR,
    RETRIEVAL_K,
)


class VectorEngine:
    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=PERSIST_DIR,
        )

    @property
    def store(self) -> Chroma:
        return self._store

    def index_pdfs(self, paths: List[str]) -> int:
        documents: List[Document] = []
        for path in paths:
            documents.extend(PyPDFLoader(path).load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        self._store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int = RETRIEVAL_K) -> List[Document]:
        return self._store.similarity_search(query, k=k)

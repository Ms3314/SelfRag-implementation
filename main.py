import glob
import os
import sys

from self_rag.config import DOCUMENTS_DIR
from self_rag.engines.vector_engine import VectorEngine
from self_rag.graph import build_graph


def index_documents() -> None:
    paths = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")))
    if not paths:
        print(f"No PDFs found in {DOCUMENTS_DIR}")
        return
    vectors = VectorEngine()
    total = vectors.index_pdfs(paths)
    print(f"Indexed {total} chunks from {len(paths)} document(s)")


def run(question: str) -> None:
    app = build_graph()
    print(f"=== Node-by-node updates for: {question} ===\n")
    for event in app.stream(
        {"query": question, "retrieval_query": question},
        stream_mode="updates",
    ):
        for node, update in event.items():
            print(f"--- {node} ---")
            print(update)
            print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        index_documents()
    elif len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        run("Who is NexaAI?")

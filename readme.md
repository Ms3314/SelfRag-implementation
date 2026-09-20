# Self-RAG Demo

A practical, from-scratch implementation of a **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) pipeline built with **LangGraph**, **Groq**, and **Chroma**. It answers questions over internal company documents, and — critically — it **checks its own work**: it decides whether retrieval is needed, filters retrieved chunks for relevance, verifies that the generated answer is grounded in the context, revises unsupported answers, and rewrites the query when an answer proves unhelpful.

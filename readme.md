# Self-RAG Demo

A practical, from-scratch implementation of a **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) pipeline built with **LangGraph**, **Groq**, and **Chroma**. It answers questions over internal company documents, and — critically — it **checks its own work**: it decides whether retrieval is needed, filters retrieved chunks for relevance, verifies that the generated answer is grounded in the context, revises unsupported answers, and rewrites the query when an answer proves unhelpful.


<h2>LangGraph Implementation</h2>

<img width="400" height="428" alt="image" src="https://github.com/user-attachments/assets/d24a0bcc-52f3-4363-b1bb-3e652f77e3a1" />

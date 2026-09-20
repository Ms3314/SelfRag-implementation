# Self-RAG Demo

A practical, from-scratch implementation of a **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) pipeline built with **LangGraph**, **Groq**, and **Chroma**. It answers questions over internal company documents, and — critically — it **checks its own work**: it decides whether retrieval is needed, filters retrieved chunks for relevance, verifies that the generated answer is grounded in the context, revises unsupported answers, and rewrites the query when an answer proves unhelpful.

Self RAG is a subconscious being which does 4 checks : 

<img width="800" height="682" alt="image" src="https://github.com/user-attachments/assets/488b4325-8350-4855-9abe-899e9bf49a71" />


<h2>LangGraph Implementation</h2>

<img width="800" height="855" alt="image" src="https://github.com/user-attachments/assets/d24a0bcc-52f3-4363-b1bb-3e652f77e3a1" />

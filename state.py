from typing import List, Literal

from pydantic import BaseModel

from langchain_core.documents import Document


class State(BaseModel):
    query: str
    retrieval_query: str = ""
    requires_retrieval: bool = False
    docs: List[Document] = []
    answer: str = ""
    relevant_docs: List[Document] = []
    context: str = ""
    issup: Literal["fully_supported", "partially_supported", "no_support"] = "no_support"
    evidence: List[str] = []
    retries: int = 0
    issue: Literal["useful", "not_useful"] = "useful"
    use_reason: str = ""
    rewrite_retries: int = 0

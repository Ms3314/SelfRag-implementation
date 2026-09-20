from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.engines.llm_engine import LLMEngine
from self_rag.engines.vector_engine import VectorEngine
from self_rag.prompts import CHECK_RELEVANCE_PROMPT, DECIDE_RETRIEVAL_PROMPT
from self_rag.schemas import RelevanceDecision, RetrieveDecision
from self_rag.state import State


class RetrievalAgent:
    def __init__(self, llm: LLMEngine, vectors: VectorEngine) -> None:
        self._llm = llm
        self._vectors = vectors

    def decide(self, state: State) -> dict:
        decision: RetrieveDecision = self._llm.structured(
            RetrieveDecision,
            [
                SystemMessage(content=DECIDE_RETRIEVAL_PROMPT),
                HumanMessage(content=f"the question is {state.query}"),
            ],
        )
        return {"requires_retrieval": decision.requires_retrieval}

    def route_decision(self, state: State) -> str:
        return "retrieve" if state.requires_retrieval else "answer_direct"

    def retrieve(self, state: State) -> dict:
        query = state.retrieval_query or state.query
        return {"docs": self._vectors.search(query)}

    def filter_relevant(self, state: State) -> dict:
        relevant_docs: List[Document] = []
        for doc in state.docs:
            decision: RelevanceDecision = self._llm.structured(
                RelevanceDecision,
                [
                    SystemMessage(content=CHECK_RELEVANCE_PROMPT),
                    HumanMessage(
                        content=f"the question is {state.query} \n\n Document : \n {doc.page_content}"
                    ),
                ],
            )
            if decision.is_relevant:
                relevant_docs.append(doc)
        return {"relevant_docs": relevant_docs}

    def route_relevance(self, state: State) -> str:
        return "generate" if state.relevant_docs else "no_answer"

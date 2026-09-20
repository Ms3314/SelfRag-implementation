from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.engines.llm_engine import LLMEngine
from self_rag.prompts import REWRITE_PROMPT
from self_rag.schemas import RewriteQueryOutput
from self_rag.state import State


class RewriteAgent:
    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def rewrite(self, state: State) -> dict:
        out: RewriteQueryOutput = self._llm.structured(
            RewriteQueryOutput,
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(
                    content=(
                        f"Questions : \n {state.query} \n\n"
                        f"Previous retrieval query : \n {state.retrieval_query} \n\n"
                        f"The Answer (if any) \n {state.answer}"
                    )
                ),
            ],
        )
        return {
            "retrieval_query": out.retrieval_query,
            "rewrite_retries": state.rewrite_retries + 1,
            "retries": 0,
            "docs": [],
            "relevant_docs": [],
            "context": "",
        }

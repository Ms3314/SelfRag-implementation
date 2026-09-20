from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.config import MAX_RETRIES, MAX_REWRITE_TRIES
from self_rag.engines.llm_engine import LLMEngine
from self_rag.prompts import SUPPORT_PROMPT, USEFULNESS_PROMPT
from self_rag.schemas import SupportDecision, UsefulnessDecision
from self_rag.state import State


class VerificationAgent:
    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def is_sup(self, state: State) -> dict:
        decision: SupportDecision = self._llm.structured(
            SupportDecision,
            [
                SystemMessage(content=SUPPORT_PROMPT),
                HumanMessage(
                    content=f"Answer : \n {state.answer} \n\n Context: \n {state.context}"
                ),
            ],
        )
        return {"issup": decision.issup, "evidence": decision.evidence}

    def route_after_issup(self, state: State) -> str:
        if state.issup == "fully_supported":
            return "to_usefulness"
        if state.retries >= MAX_RETRIES:
            return "to_usefulness"
        return "revise"

    def is_useful(self, state: State) -> dict:
        decision: UsefulnessDecision = self._llm.structured(
            UsefulnessDecision,
            [
                SystemMessage(content=USEFULNESS_PROMPT),
                HumanMessage(
                    content=f"the Question is \n {state.query} \n\n The Answer is : \n {state.answer}"
                ),
            ],
        )
        return {"issue": decision.issue, "use_reason": decision.reason}

    def route_usefulness(self, state: State) -> str:
        if state.issue == "useful":
            return "useful"
        if state.rewrite_retries >= MAX_REWRITE_TRIES:
            return "no_answer_found"
        return "rewrite_question"

from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.config import MAX_RETRIES
from self_rag.engines.llm_engine import LLMEngine
from self_rag.prompts import DIRECT_PROMPT, GENERATE_PROMPT, REVISER_PROMPT
from self_rag.state import State

FALLBACK_PREFIX = (
    "I can't answer this question directly, but here is what I found that might be relevant:"
)


class GenerationAgent:
    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def answer_direct(self, state: State) -> dict:
        answer = self._llm.text(
            [
                SystemMessage(content=DIRECT_PROMPT),
                HumanMessage(content=f"the query is {state.query}"),
            ]
        )
        return {"answer": answer}

    def generate_from_context(self, state: State) -> dict:
        context = "\n\n----\n\n".join(
            doc.page_content for doc in state.relevant_docs
        ).strip()

        if not context:
            return {"answer": "no relevant information provided", "context": ""}

        answer = self._llm.text(
            [
                SystemMessage(content=GENERATE_PROMPT),
                HumanMessage(
                    content=f"query is : {state.query} \n\n with context as \n\n {context}"
                ),
            ]
        )
        return {"answer": answer, "context": context}

    def revise(self, state: State) -> dict:
        new_answer = self._llm.text(
            [
                SystemMessage(content=REVISER_PROMPT),
                HumanMessage(
                    content=(
                        "QUESTION:\n"
                        f"<question>\n{state.query}\n</question>\n\n"
                        "ANSWER:\n"
                        f"<answer>\n{state.answer}\n</answer>\n\n"
                        "CONTEXT:\n"
                        f"<context>\n{state.context}\n</context>\n"
                    )
                ),
            ]
        )

        retries = state.retries + 1
        if new_answer.strip() == state.answer.strip():
            retries = MAX_RETRIES

        return {"answer": new_answer, "retries": retries}

    def no_answer_found(self, state: State) -> dict:
        sources = state.relevant_docs or state.docs
        if not sources:
            return {"answer": "Sorry, I don't know the answer to this question"}

        seen = set()
        snippets = []
        for doc in sources:
            text = doc.page_content.strip()
            if text and text not in seen:
                seen.add(text)
                snippets.append(text)

        return {"answer": f"{FALLBACK_PREFIX}\n\n" + "\n\n".join(snippets)}

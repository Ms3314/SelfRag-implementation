from typing import List, Type, TypeVar

from pydantic import BaseModel

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from self_rag.config import CHAT_MODEL

T = TypeVar("T", bound=BaseModel)


class LLMEngine:
    def __init__(self, model: str = CHAT_MODEL):
        self.model = model
        self._llm = ChatGroq(
            model=model,
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )

    def text(self, messages: List[BaseMessage]) -> str:
        return self._llm.invoke(messages).content

    def structured(self, schema: Type[T], messages: List[BaseMessage]) -> T:
        runnable = self._llm.with_structured_output(schema, method="json_schema")
        return runnable.invoke(messages)

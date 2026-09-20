from typing import List, Literal

from pydantic import BaseModel, Field


class RetrieveDecision(BaseModel):
    requires_retrieval: bool


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the question, else False.",
    )


class SupportDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


class UsefulnessDecision(BaseModel):
    issue: Literal["useful", "not_useful"]
    reason: str = Field(description="Short reason for the usefulness or not usefulness")


class RewriteQueryOutput(BaseModel):
    retrieval_query: str = Field(
        description="Rewritten query optimized for vector retrieval against internal company PDFs"
    )

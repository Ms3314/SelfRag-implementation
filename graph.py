from langgraph.graph import END, START, StateGraph

from self_rag.agents.generation_agent import GenerationAgent
from self_rag.agents.retrieval_agent import RetrievalAgent
from self_rag.agents.rewrite_agent import RewriteAgent
from self_rag.agents.verification_agent import VerificationAgent
from self_rag.engines.llm_engine import LLMEngine
from self_rag.engines.vector_engine import VectorEngine
from self_rag.state import State


def build_graph(llm: LLMEngine | None = None, vectors: VectorEngine | None = None):
    llm = llm or LLMEngine()
    vectors = vectors or VectorEngine()

    retrieval = RetrievalAgent(llm, vectors)
    generation = GenerationAgent(llm)
    verification = VerificationAgent(llm)
    rewrite = RewriteAgent(llm)

    graph = StateGraph(State)

    graph.add_node("check_retrieval", retrieval.decide)
    graph.add_node("retrieve", retrieval.retrieve)
    graph.add_node("answer_direct", generation.answer_direct)
    graph.add_node("filter_relevant", retrieval.filter_relevant)
    graph.add_node("generate", generation.generate_from_context)
    graph.add_node("no_answer_found", generation.no_answer_found)
    graph.add_node("is_sup", verification.is_sup)
    graph.add_node("revise", generation.revise)
    graph.add_node("is_useful", verification.is_useful)
    graph.add_node("rewrite_question", rewrite.rewrite)

    graph.add_edge(START, "check_retrieval")
    graph.add_conditional_edges(
        "check_retrieval",
        retrieval.route_decision,
        {
            "retrieve": "retrieve",
            "answer_direct": "answer_direct",
        },
    )

    graph.add_edge("retrieve", "filter_relevant")
    graph.add_conditional_edges(
        "filter_relevant",
        retrieval.route_relevance,
        {
            "generate": "generate",
            "no_answer": "no_answer_found",
        },
    )

    graph.add_edge("generate", "is_sup")
    graph.add_conditional_edges(
        "is_sup",
        verification.route_after_issup,
        {
            "to_usefulness": "is_useful",
            "revise": "revise",
        },
    )

    graph.add_conditional_edges(
        "is_useful",
        verification.route_usefulness,
        {
            "useful": END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )

    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("revise", "is_sup")
    graph.add_edge("answer_direct", END)

    return graph.compile()

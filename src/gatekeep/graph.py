"""LangGraph agentic RAG whose judgment steps are delegated to interchangeable gates."""
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from gatekeep.llm import GEMMA

REFUSAL = "I can't answer that from this book."
EVIDENCE_CHARS = 2000  # fine-tuned checkpoints read 1,024 tokens (768 for the state): ~500 for evidence, the rest for the answer


class S(TypedDict, total=False):
    q: str
    query: str
    route: str
    docs: list
    answer: str
    rewrites: int
    regens: int
    retry: bool
    tried: list    # the passages of every search so far, to notice a rewrite that finds nothing new
    repeat: bool
    feedback: str
    flagged: bool
    gate_calls: int
    gate_s: float
    retr_s: float


def gen_prompt(q, docs, feedback):
    ctx = "\n\n".join(f"[{i + 1}] {d['text']}" for i, d in enumerate(docs))
    fb = f"\nA reviewer said your previous answer was flawed: {feedback}.\n" if feedback else ""
    return ("Answer the question using only the passages. If they are not enough, say you cannot answer.\n\n"
            f"{ctx}\n\nQuestion: {q}\n{fb}Answer:")


def clean_query(text, fallback):
    """The model answers a rewrite request with markdown, quotes or a chatty list; keep one short query or the question itself."""
    lines = [l.strip(" \t*_`\"'-•") for l in text.splitlines()]
    lines = [l for l in lines if l and not l.endswith(":")]  # "Here are some queries:" is chatter
    q = lines[0] if lines else ""
    return q if 0 < len(q.split()) <= 12 else fallback


def evidence(docs):
    out, used = [], 0
    for d in docs:
        if out and used + len(d["text"]) > EVIDENCE_CHARS:
            break
        out.append(d["text"][:EVIDENCE_CHARS])
        used += len(d["text"])
    return "\n\n".join(out)


def make_nodes(index, llm, gates, model, use_gates, max_rewrites, max_regen, feedback):
    def ask(s, name, texts):
        out = gates[name].decide_many(name, texts)
        return out, {"gate_calls": s.get("gate_calls", 0) + len(texts),
                     "gate_s": s.get("gate_s", 0.0) + gates[name].last_s}

    def route(s):
        if not use_gates:
            return {"route": "retrieve"}
        out, upd = ask(s, "route", [s["q"]])
        return {"route": out[0][0], **upd}

    def direct(s):
        return {"answer": llm.chat(model, s["q"], max_tokens=300)}

    def refuse(s):
        return {"answer": REFUSAL}

    def retrieve(s):
        t0 = time.perf_counter()
        query = s.get("query") or s["q"]
        docs = index.rerank(query, index.search(query, 20), 5)
        key, seen = sorted(d["text"] for d in docs), s.get("tried", [])
        repeat = key in seen  # grading the same passages again would only repeat the same verdict
        return {"docs": [] if repeat else docs, "tried": seen + [key], "repeat": repeat,
                "retr_s": s.get("retr_s", 0.0) + time.perf_counter() - t0}

    def grade(s):
        if not use_gates:
            return {"docs": s["docs"]}  # a node must write something
        out, upd = ask(s, "grade", [f"Question: {s['q']}\nPassage: {d['text']}" for d in s["docs"]])
        return {"docs": [d for d, (label, _) in zip(s["docs"], out) if label == "yes"], **upd}

    def rewrite(s):
        q = llm.chat(model, f"Rewrite as a short search query for a machine-learning textbook: {s['q']}", max_tokens=40)
        return {"query": clean_query(q, s["q"]), "rewrites": s.get("rewrites", 0) + 1}

    def generate(s):
        temp = 0.7 if s.get("regens") else 0.0  # a blind retry at temperature 0 would repeat itself
        return {"answer": llm.chat(model, gen_prompt(s["q"], s["docs"], s.get("feedback", "")),
                                   max_tokens=400, temperature=temp)}

    def check(s):
        if not use_gates:
            return {"retry": False}
        g, u1 = ask(s, "grounded", [f"Passages:\n{evidence(s['docs'])}\n\nAnswer: {s['answer']}"])
        n, u2 = ask({**s, **u1}, "sufficient", [f"Question: {s['q']}\nAnswer: {s['answer']}"])
        ungrounded, insufficient = g[0][0] == "no", n[0][0] == "no"
        if not (ungrounded or insufficient):
            return {**u2, "retry": False, "feedback": ""}
        regens = s.get("regens", 0) + 1
        why = "not supported by the passages" if ungrounded else "does not address the question"
        retry = regens <= max_regen
        return {**u2, "regens": regens, "retry": retry, "flagged": not retry,
                "feedback": why if feedback else ""}

    def after_route(s):
        return {"retrieve": "retrieve", "direct": "direct", "off_topic": "refuse"}[s["route"]]

    def after_retrieve(s):
        return "refuse" if s.get("repeat") else "grade"

    def after_grade(s):
        if s["docs"]:
            return "generate"
        return "rewrite" if s.get("rewrites", 0) < max_rewrites else "refuse"

    def after_check(s):
        return "generate" if s.get("retry") else END

    nodes = dict(route=route, direct=direct, refuse=refuse, retrieve=retrieve, grade=grade,
                 rewrite=rewrite, generate=generate, check=check)
    return nodes, after_route, after_retrieve, after_grade, after_check


def build_graph(index, llm, gates, model=GEMMA, use_gates=True, max_rewrites=2, max_regen=2, feedback=False):
    n, after_route, after_retrieve, after_grade, after_check = make_nodes(
        index, llm, gates, model, use_gates, max_rewrites, max_regen, feedback)
    g = StateGraph(S)
    for name, fn in n.items():
        g.add_node(name, fn)
    g.set_entry_point("route")
    g.add_conditional_edges("route", after_route, {"retrieve": "retrieve", "direct": "direct", "refuse": "refuse"})
    g.add_conditional_edges("retrieve", after_retrieve, {"grade": "grade", "refuse": "refuse"})
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, {"generate": "generate", END: END})
    g.add_edge("direct", END)
    g.add_edge("refuse", END)
    return g.compile()


def build_agents_graph(index, llm, gates, model=GEMMA, use_gates=True, max_rewrites=2, max_regen=2):
    """Same logic as build_graph plus critic feedback to the writer, wired as Researcher/Writer/Critic subgraphs."""
    n, after_route, after_retrieve, _, after_check = make_nodes(
        index, llm, gates, model, use_gates, max_rewrites, max_regen, feedback=True)

    researcher = StateGraph(S)
    for name in ("retrieve", "grade", "rewrite"):
        researcher.add_node(name, n[name])
    researcher.set_entry_point("retrieve")
    researcher.add_conditional_edges("retrieve", after_retrieve, {"grade": "grade", "refuse": END})
    researcher.add_conditional_edges(
        "grade", lambda s: "rewrite" if not s["docs"] and s.get("rewrites", 0) < max_rewrites else END,
        {"rewrite": "rewrite", END: END})
    researcher.add_edge("rewrite", "retrieve")

    writer = StateGraph(S)
    writer.add_node("generate", n["generate"])
    writer.set_entry_point("generate")
    writer.add_edge("generate", END)

    critic = StateGraph(S)
    critic.add_node("check", n["check"])
    critic.set_entry_point("check")
    critic.add_edge("check", END)

    g = StateGraph(S)
    for name in ("route", "direct", "refuse"):
        g.add_node(name, n[name])
    g.add_node("researcher", researcher.compile())
    g.add_node("writer", writer.compile())
    g.add_node("critic", critic.compile())
    g.set_entry_point("route")
    g.add_conditional_edges("route", after_route, {"retrieve": "researcher", "direct": "direct", "refuse": "refuse"})
    g.add_conditional_edges("researcher", lambda s: "writer" if s["docs"] else "refuse",
                            {"writer": "writer", "refuse": "refuse"})
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", after_check, {"generate": "writer", END: END})
    g.add_edge("direct", END)
    g.add_edge("refuse", END)
    return g.compile()

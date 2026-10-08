"""Run one question through the gates and yield each step as an event, for the trace and the web app."""
import time

from gatekeep.gates import GATES, Cascade, Gate, LLMGate
from gatekeep.graph import REFUSAL, build_graph


class Recorder(Gate):
    """Wraps a gate backend and notes every decision (who decided, how long it took) for the trace."""
    def __init__(self, inner, log):
        self.inner, self.log = inner, log

    def decide_many(self, gate, texts):
        e0, t0 = getattr(self.inner, "escalated", 0), time.perf_counter()
        out = self.inner.decide_many(gate, texts)
        self.last_s = self.inner.last_s
        self.log.append({"gate": gate, "labels": [lab for lab, _ in out], "n": len(texts),
                         "escalated": getattr(self.inner, "escalated", 0) - e0,
                         "ms": round(1000 * (time.perf_counter() - t0))})
        return out


# "Needed to answer" fails every passage of "what is bagging and boosting?": each one covers a single half.
LENIENT_GRADE = "Does the passage help answer any part of the question, even if it covers only one part?"


class AllNoFallback(Gate):
    """Fast mode's grade gate: Laya decides, but when it rejects every passage Gemma re-checks them once.
    Laya's grade gate learned "this passage alone answers the question", so it rejects every passage of a broad
    or comparison question; the measured V2 has no such fallback (see docs/RESULTS.md)."""
    def __init__(self, fast, slow):
        self.fast, self.slow, self.escalated = fast, slow, 0

    def decide_many(self, gate, texts):
        out = self.fast.decide_many(gate, texts)
        self.last_s = self.fast.last_s
        if len(texts) > 1 and all(label == "no" for label, _ in out):
            out = self.slow.decide_many(gate, texts)
            self.escalated += len(texts)
        return out


def who(e):
    return "Laya" if not e["escalated"] else f"Laya, then Gemma for {e['escalated']} of {e['n']}"


def describe(node, out):
    if node == "route":
        return f"route -> {out['route']}"
    if node == "retrieve":
        return f"retrieve: top {len(out['docs'])} passages after reranking"
    if node == "grade":
        return f"grade: kept {len(out['docs'])} passages"
    if node == "rewrite":
        return f"rewrite the search query: {out['query']!r}"
    if node == "generate":
        return "generate: the writer drafts an answer"
    if node == "check":
        return "check: " + ("the answer passed" if not out.get("retry") and not out.get("flagged")
                            else "flagged, no more retries" if out.get("flagged")
                            else f"failed ({out['feedback'] or 'a gate said no'}), regenerating")
    return node


def _gates(mode, laya, llm, taus, log):
    if mode == "fast":   # V2 (every gate by Laya), plus a Gemma re-check when Laya rejects every passage
        recheck = LLMGate(llm, overrides={"grade": LENIENT_GRADE})
        return {g: Recorder(AllNoFallback(laya, recheck) if g == "grade" else laya, log) for g in GATES}
    if mode == "careful":  # V3: Laya first, Gemma when Laya is unsure; Gemma judges passages with the lenient question
        judge = {g: LLMGate(llm, overrides={"grade": LENIENT_GRADE} if g == "grade" else None) for g in GATES}
        return {g: Recorder(Cascade(laya, judge[g], taus[g]), log) for g in GATES}
    raise ValueError(f"unknown mode {mode!r}: use 'fast', 'careful' or 'auto'")


def ask_events(question, index, llm, laya, taus, mode="careful"):
    """Run one question and yield what happens, step by step, for the trace and the web app."""
    if mode == "auto":
        yield from auto_events(question, index, llm, laya, taus)
        return
    log, t0 = [], time.perf_counter()
    app = build_graph(index, llm, _gates(mode, laya, llm, taus, log))
    docs, answer, route, flagged = [], "", None, False
    for update in app.stream({"q": question}, stream_mode="updates"):
        for node, out in update.items():
            yield {"type": "step", "node": node, "text": describe(node, out)}
            for e in log:
                yield {"type": "gate", "gate": e["gate"], "labels": e["labels"], "n": e["n"], "who": who(e), "ms": e["ms"]}
            log.clear()
            docs, answer, route = out.get("docs", docs), out.get("answer", answer), out.get("route", route)
            flagged = out.get("flagged", flagged)
    refused = answer == REFUSAL
    yield {"type": "passages", "items": [{"text": d["text"], "section": d.get("section", ""), "page": d.get("page"),
                                          "source": d.get("source", "")} for d in ([] if refused else docs)]}
    yield {"type": "answer", "text": answer, "refused": refused, "flagged": flagged,
           "reason": ("off_topic" if route == "off_topic" else "no_passage") if refused else None}
    yield {"type": "done", "seconds": round(time.perf_counter() - t0, 2)}


def auto_events(question, index, llm, laya, taus):
    """Fast first; if it finds no passage or can only offer a flagged answer, run Careful and show that result instead."""
    t0, held = time.perf_counter(), []

    def live(events):  # steps and gate decisions stream as they happen; the passages and answer wait until they are final
        for ev in events:
            if ev["type"] in ("passages", "answer", "done"):
                held.append(ev)
            else:
                yield ev

    yield from live(ask_events(question, index, llm, laya, taus, "fast"))
    if held[1]["reason"] == "no_passage" or held[1]["flagged"]:
        held.clear()
        yield {"type": "step", "node": "escalate", "text": "Fast could not answer this, so Gemma takes over and judges every passage"}
        yield from live(ask_events(question, index, llm, laya, taus, "careful"))
    yield from held[:2]
    yield {"type": "done", "seconds": round(time.perf_counter() - t0, 2)}

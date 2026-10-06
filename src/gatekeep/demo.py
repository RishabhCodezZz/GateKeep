"""Interactive demo: ask a question and watch the route, the passages kept, each gate decision and the answer."""
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


def gate_line(e):
    who = "Laya" if not e["escalated"] else f"Laya, then Gemma for {e['escalated']} of {e['n']}"
    labels = e["labels"][0] if e["n"] == 1 else f"{e['labels'].count('yes')} of {e['n']} yes"
    return f"   {e['gate']}: {labels} ({who}, {e['ms']} ms)"


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


def ask(question, index, llm, laya, taus):
    """Run one question through the V3 cascade. Returns (answer, trace lines, passages, seconds)."""
    log = []
    gates = {g: Recorder(Cascade(laya, LLMGate(llm), taus[g]), log) for g in GATES}
    app = build_graph(index, llm, gates)
    trace, docs, answer, t0 = [], [], "", time.perf_counter()
    for update in app.stream({"q": question}, stream_mode="updates"):
        for node, out in update.items():
            trace.append(describe(node, out))
            trace += [gate_line(e) for e in log]
            log.clear()
            docs = out.get("docs", docs)
            answer = out.get("answer", answer)
    if answer == REFUSAL:  # the pipeline's fixed message says "book"; this demo answers from the scikit-learn user guide
        answer = "I can't answer that from the scikit-learn user guide."
    return answer, trace, [d["text"] for d in docs], time.perf_counter() - t0


def launch(index, llm, laya, taus, examples=()):
    import gradio as gr

    def run(question):
        answer, trace, passages, secs = ask(question, index, llm, laya, taus)
        shown = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "(no passage was kept)"
        return answer, "\n".join(trace) + f"\n\ntotal {secs:.1f} s", shown

    gr.Interface(run, gr.Textbox(label="Your question", lines=2),
                 [gr.Textbox(label="Answer"), gr.Textbox(label="What the gates did", lines=14),
                  gr.Textbox(label="Passages the answer was based on", lines=10)],
                 examples=[[e] for e in examples], title="GateKeep",
                 description="Fine-tuned Laya makes the gate decisions; Gemma steps in when Laya is unsure. "
                             "Answers come from the scikit-learn user guide.").launch(share=True)

"""Run a variant over a question set, judge the answers, summarise with confidence intervals."""
import random

from gatekeep.gates import GATES, Cascade, LLMGate
from gatekeep.graph import REFUSAL, build_graph
from gatekeep.llm import SUPER, ULTRA
from gatekeep.metrics import bootstrap_ci


def make_app(variant, index, llm, laya=None, tau=0.8, agents=False):
    llm_gate = LLMGate(llm)
    if variant == "V0":
        gates = {}
    elif variant == "V1":
        gates = {g: llm_gate for g in GATES}
    elif variant == "V2":
        gates = {g: laya for g in GATES}
    elif variant == "V3":
        gates = {g: Cascade(laya, llm_gate, tau[g] if isinstance(tau, dict) else tau) for g in GATES}
    else:
        raise ValueError(variant)
    if agents:
        from gatekeep.graph import build_agents_graph as build
    else:
        build = build_graph
    return build(index, llm, gates, use_gates=(variant != "V0")), llm_gate


def correct_prompt(item, ans):
    if item["answerable"]:
        return (f"Question: {item['q']}\nReference answer: {item['a']}\nCandidate answer: {ans}\n"
                "Does the candidate agree with the reference answer? Reply CORRECT or INCORRECT.")
    return (f"Question: {item['q']}\nThis question cannot be answered from the textbook.\nCandidate answer: {ans}\n"
            "Does the candidate decline to answer? Reply CORRECT if it declines, INCORRECT if it answers anyway.")


def is_correct(text):
    return int(text.strip().upper().startswith("CORRECT"))


def judge(llm, item, result, model=SUPER):
    ans = result["answer"]
    correct = is_correct(llm.chat(model, correct_prompt(item, ans), max_tokens=8))
    docs = result.get("docs") or []
    if ans == REFUSAL or not docs:
        faithful = None
    else:
        # the grader reads every passage the writer saw; the 1,200-char evidence cap is only for Laya's 512-token window
        passages = "\n\n".join(d["text"] for d in docs)
        v = llm.chat(model, f"Passages:\n{passages}\n\nAnswer: {ans}\n"
                            "Is every claim in the answer supported by the passages? Reply YES or NO.", max_tokens=8)
        faithful = int(v.strip().upper().startswith("YES"))
    return {"correct": correct, "faithful": faithful}


def agreement(llm, items, rows, n=200, seed=0):
    """Share of answers where Super and Ultra give the same correctness verdict: the validity check for Super as grader."""
    by_id = {it["id"]: it for it in items}
    pick = random.Random(seed).sample(rows, min(n, len(rows)))
    same = sum(is_correct(llm.chat(SUPER, correct_prompt(by_id[r["id"]], r["answer"]), max_tokens=8))
               == is_correct(llm.chat(ULTRA, correct_prompt(by_id[r["id"]], r["answer"]), max_tokens=8)) for r in pick)
    return same / len(pick)


def run_variant(app, llm_gate, llm, items):
    rows = []
    for it in items:
        c0, s0, g0 = llm.logical_calls, llm.total_s, llm_gate.llm_used
        r = app.invoke({"q": it["q"]})
        # measure the system BEFORE judging, so grading calls are not charged to it
        row = {"id": it["id"], "answerable": it["answerable"], "answer": r["answer"],
               "llm_calls": llm.logical_calls - c0, "gate_llm_calls": llm_gate.llm_used - g0,
               "gate_calls": r.get("gate_calls", 0),
               "latency_s": (llm.total_s - s0) + r.get("gate_s", 0.0) + r.get("retr_s", 0.0),
               "flagged": bool(r.get("flagged")), "route": r.get("route"), "rewrites": r.get("rewrites", 0),
               "regens": r.get("regens", 0), "kept": len(r.get("docs") or [])}
        rows.append({**row, **judge(llm, it, r)})
    return rows


def summarize(rows):
    out = {}
    for k in ("correct", "faithful", "llm_calls", "gate_llm_calls", "latency_s"):
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = (sum(v) / len(v), *bootstrap_ci(v)) if v else (float("nan"),) * 3
    return out

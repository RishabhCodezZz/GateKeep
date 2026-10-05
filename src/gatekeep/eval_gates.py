"""Per-gate evaluation of every backend, plus the retrieval checks and the Laya-vs-cross-encoder reranking ablation."""
import time

import numpy as np

from gatekeep.gates import LLMGate
from gatekeep.metrics import ece, macro_f1


def evaluate_gate(backend, gate, rows, llm=None):
    preds, secs = [], []
    for r in rows:
        s0, t0 = (llm.total_s if llm else 0.0), time.perf_counter()
        preds.append(backend.decide_many(gate, [r["text"]])[0])
        secs.append(llm.total_s - s0 if isinstance(backend, LLMGate) else time.perf_counter() - t0)
    y = [r["label"] for r in rows]
    labels = [p[0] for p in preds]
    ok = [a == b for a, b in zip(labels, y)]
    conf = [p[1] for p in preds]
    out = {"f1": macro_f1(y, labels), "acc": float(np.mean(ok)), "ece": ece(conf, ok),
           "p50_ms": 1000 * float(np.percentile(secs, 50)), "p95_ms": 1000 * float(np.percentile(secs, 95)),
           "ok": ok, "conf": conf, "recall_yes": None, "precision_yes": None}
    if set(y) <= {"yes", "no"}:  # the three yes/no gates: how many relevant/grounded/sufficient cases does it let through?
        tp = sum(a == "yes" and b == "yes" for a, b in zip(labels, y))
        out["recall_yes"] = tp / max(y.count("yes"), 1)
        out["precision_yes"] = tp / labels.count("yes") if "yes" in labels else None
    return out


def dev_acc_from_csv(path, gate, backend):
    """A backend's dev accuracy for one gate, from an earlier evalgates run (so the LLM need not be re-run)."""
    import csv
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if (r["gate"], r["backend"], r["split"]) == (gate, backend, "dev"):
            return float(r["acc"])
    raise KeyError((gate, backend))


def retrieval_recall(index, items):
    """How often the gold chunk is found by the search (top 20), and how often it survives the reranker (top 5).
    A gate cannot fix a passage that was never retrieved, so this separates retrieval misses from gate mistakes."""
    s20 = s5 = n = 0
    for it in items:
        if not it["answerable"] or not it["gold"]:
            continue
        gold = set(it["gold"])
        hits = index.search(it["q"], 20)
        top5 = index.rerank(it["q"], hits, 5)
        s20 += bool(gold & {h["id"] for h in hits})
        s5 += bool(gold & {h["id"] for h in top5})
        n += 1
    return {"n": n, "in_search_top20": s20 / n, "in_rerank_top5": s5 / n}


def hit_at_k(index, laya, items, k=5):
    ce = la = n = 0
    for it in items:
        if not it["answerable"]:
            continue
        hits, gold = index.search(it["q"], 20), set(it["gold"])
        by_ce = index.rerank(it["q"], hits, k)
        sc = laya.decide_many("grade", [f"Question: {it['q']}\nPassage: {h['text']}" for h in hits])
        ps = [c if label == "yes" else 1 - c for label, c in sc]  # P(relevant)
        by_laya = [h for _, h in sorted(zip(ps, hits), key=lambda t: -t[0])][:k]
        ce += bool(gold & {h["id"] for h in by_ce})
        la += bool(gold & {h["id"] for h in by_laya})
        n += 1
    return {"cross_encoder": ce / n, "laya": la / n}

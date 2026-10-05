"""Training/eval rows for the four gates. Labels come from the book's own structure or from Super.

Lessons from the first per-gate run (2026-10-05):
- Gate 4 rows whose "no" side was Super's evasive text were solvable at 98% by word counting alone (a style giveaway),
  so both sides are now Gemma-written and the "no" is Gemma's answer to a different question.
- Gate 3 is trained on the same kind of rows it is tested on: real Gemma answers, labeled by Super.
"""
import json
import random
from pathlib import Path

from gatekeep.gates import GATES
from gatekeep.graph import evidence, gen_prompt
from gatekeep.llm import GEMMA, SUPER
from gatekeep.qa import parse_json


def _yes(text):
    return text.strip().upper().startswith("YES")


def gen_direct(llm, n=40):
    j = parse_json(llm.chat(SUPER, f"Write {n} short messages that need no document lookup: greetings, thanks, "
                                   "'what is 2+2'-style trivia. JSON only: {\"messages\": [\"...\"]}", max_tokens=3000))
    return (j or {}).get("messages", [])


def g1_rows(items, direct_texts):
    rows = [{"text": it["q"], "label": "retrieve" if it["answerable"] else "out_of_scope"} for it in items]
    return rows + [{"text": t, "label": "direct"} for t in direct_texts]


def g2_rows(llm, index, items, by_id, negs=2):
    rows = []
    for it in items:
        if not it["answerable"]:
            continue
        gold = set(it["gold"])
        rows += [{"text": f"Question: {it['q']}\nPassage: {by_id[c]['text']}", "label": "yes"} for c in it["gold"]]
        for h in [h for h in index.search(it["q"], 20) if h["id"] not in gold][:negs]:
            ans = llm.chat(SUPER, f"Question: {it['q']}\nPassage: {h['text']}\nDoes the passage contain information "
                                  "needed to answer the question? Reply YES or NO.", max_tokens=8)
            rows.append({"text": f"Question: {it['q']}\nPassage: {h['text']}", "label": "yes" if _yes(ans) else "no"})
    return rows


def g3_rows(llm, index, items, by_id, temps=(0.0,)):
    """Gemma answers from the passages the system really retrieves; Super says whether every claim is supported.
    More than one temperature gives more (different) answers per question."""
    rows = []
    for it in items:
        if not it["answerable"]:
            continue
        docs = index.rerank(it["q"], index.search(it["q"], 20), 5)
        for t in temps:
            answer = llm.chat(GEMMA, gen_prompt(it["q"], docs, ""), max_tokens=300, temperature=t)
            ev = f"Passages:\n{evidence(docs)}\n\nAnswer: "
            v = llm.chat(SUPER, f"{ev}{answer}\nIs every claim in the answer supported by the passages? Reply YES or NO.",
                         max_tokens=8)
            rows.append({"text": ev + answer, "label": "yes" if _yes(v) else "no"})
    return rows


def g4_rows(llm, items, by_id, seed=0):
    """'yes': Gemma's answer to the question. 'no': Gemma's answer to a different question from the same chapter.
    Both sides are written the same way, so only meaning can tell them apart."""
    ok = [it for it in items if it["answerable"]]
    answers = {it["id"]: llm.chat(GEMMA, gen_prompt(it["q"], [by_id[c] for c in it["gold"]], ""), max_tokens=300) for it in ok}
    rng, rows = random.Random(seed), []
    for it in ok:
        others = [o for o in ok if o["id"] != it["id"] and o["chapter"] == it["chapter"]] or [o for o in ok if o["id"] != it["id"]]
        if not others:
            continue
        other = rng.choice(others)
        rows += [{"text": f"Question: {it['q']}\nAnswer: {answers[it['id']]}", "label": "yes"},
                 {"text": f"Question: {it['q']}\nAnswer: {answers[other['id']]}", "label": "no"}]
    return rows


def gold_for(gate, label):
    """The training target Laya's official fine-tuning notebook reads: a probability per option
    ("true"/"false" for yes/no gates, one entry per criterion for the route gate)."""
    q = GATES[gate]["q"]
    if q["type"] == "noul":
        return {"probabilities": {"true": float(label == "yes"), "false": float(label != "yes")}}
    return {"probabilities": {k: float(k == label) for k in q["criteria"]}}


def balance_rows(rows, min_share=0.25):
    """Repeat the rarer labels until each has at least min_share of the most common label's count (training rows only)."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    top = max(len(v) for v in by_label.values())
    want = round(top * min_share)
    out = list(rows)
    for v in by_label.values():
        if len(v) < want:
            out += [v[i % len(v)] for i in range(want - len(v))]
    return out


def write_gate_data(rows_by_gate, split, outdir="data/gates"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for gate, rows in rows_by_gate.items():
        with open(f"{outdir}/{gate}_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

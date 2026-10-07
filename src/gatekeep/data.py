"""Training/eval rows for the four gates. Labels come from the book's own structure or from Super.

Lessons from the first per-gate run (2026-10-05):
- Gate 4 rows whose "no" side was Super's evasive text were solvable at 98% by word counting alone (a style giveaway),
  so both sides are now Gemma-written and the "no" is Gemma's answer to a different question.
- Gate 3 is trained on the same kind of rows it is tested on: real Gemma answers, labeled by Super.
"""
import json
import random
import re
from pathlib import Path

from gatekeep.gates import GATES
from gatekeep.graph import evidence, gen_prompt
from gatekeep.llm import GEMMA, SUPER
from gatekeep.qa import parse_json


def _yes(text):
    return text.strip().upper().startswith("YES")


DIRECT_KINDS = ["greetings", "thanks and goodbyes", "small talk about the assistant itself",
                "trivial arithmetic or unit conversions", "very general questions about the weather or the time",
                "requests to repeat or rephrase something"]
_LIST_MARK = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")


ML_SHORT_KINDS = ["one-line 'what is X' questions about basic machine-learning terms",
                  "short how-to questions about training or evaluating a model",
                  "questions a beginner would type about neural networks or deep learning",
                  "short questions about data preparation, features or metrics",
                  "short questions about recent ML topics such as LLMs, LoRA, RAG or diffusion models"]
OFF_TOPIC_KINDS = ["geography and travel", "cooking and food", "sports", "history and politics",
                   "films, music and celebrities", "health and everyday life", "shopping and personal finance"]


def gen_messages(llm, kinds, n, rule):
    """Short user messages, one per line. The first run asked for JSON, got something unparseable,
    and silently returned nothing, so the 'direct' class vanished from gate 1."""
    per = -(-n // len(kinds))
    out = []
    for kind in kinds:
        text = llm.chat(SUPER, f"Write {per} different short messages a user might send to an assistant: {kind}. "
                               f"{rule} One message per line, no numbering, no quotes.",
                        max_tokens=800, temperature=0.7)
        for line in text.splitlines():
            m = _LIST_MARK.sub("", line).strip().strip("\"'")
            if 2 <= len(m) <= 120 and m not in out:
                out.append(m)
    return out[:n]


def gen_direct(llm, n=120):
    return gen_messages(llm, DIRECT_KINDS, n, "They must not be about machine learning.")  # same prompt as v1: cached


def gen_ml_short(llm, n=160):
    return gen_messages(llm, ML_SHORT_KINDS, n, "They must be short questions about machine learning.")


def gen_off_topic(llm, n=160):
    return gen_messages(llm, OFF_TOPIC_KINDS, n, "They must not be about machine learning, data or programming.")


def split_messages(msgs, seed=0, dev=0.15, test=0.25):
    """Shuffle, then cut: v1 cut in generation order, so each split held different kinds of message."""
    m = list(msgs)
    random.Random(seed).shuffle(m)
    a, b = round(len(m) * (1 - dev - test)), round(len(m) * (1 - test))
    return {"train": m[:a], "dev": m[a:b], "test": m[b:]}


def g1_rows(items, direct, general=(), off_topic=()):
    """Route rows. Every book question is 'retrieve', answerable or not: whether the book covers a topic is
    decided after retrieval by the grade gate, never guessed from the question (v1's mistake)."""
    pairs = (("retrieve", [it["q"] for it in items]), ("retrieve", general), ("off_topic", off_topic), ("direct", direct))
    return [{"text": t, "label": label} for label, texts in pairs for t in texts]


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


SMOOTH = 0.1  # the official recipe trains on probabilities; v1's hard 1/0 targets made grade and sufficient overconfident


def _options(gate):
    q = GATES[gate]["q"]
    return ["false", "true"] if q["type"] == "noul" else list(q["criteria"])


def _hit(gate, label):
    hit = {"yes": "true", "no": "false"}.get(label, label) if GATES[gate]["q"]["type"] == "noul" else label
    if hit not in _options(gate):  # a stale label (v1's out_of_scope) would silently give an empty target
        raise ValueError(f"label {label!r} is not an option of gate {gate!r}: {_options(gate)}")
    return hit


def gold_for(gate, label):
    """The training target Laya's official fine-tuning notebook reads: a probability per option."""
    keys, hit = _options(gate), _hit(gate, label)
    return {"probabilities": {k: round(1 - SMOOTH if k == hit else SMOOTH / (len(keys) - 1), 6) for k in keys}}


def target_vector(gate, label):
    """One-hot target in Laya's option order, for calibration records."""
    hit = _hit(gate, label)
    return [float(k == hit) for k in _options(gate)]


def write_gate_data(rows_by_gate, split, outdir="data/gates"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for gate, rows in rows_by_gate.items():
        with open(f"{outdir}/{gate}_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

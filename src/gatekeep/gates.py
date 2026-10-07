"""The four gates and the interchangeable backends that answer them."""
import os, re, time

from gatekeep.llm import GEMMA

GATES = {
    "route": {"q": {"type": "choice",
                    "instructions": "Is this message a machine-learning question to look up, small talk, or about something else?",
                    "criteria": {"retrieve": "a question about machine learning, data or ML programming, even if the textbook may not cover it",
                                 "direct": "greeting, thanks or trivial arithmetic that needs no lookup",
                                 "off_topic": "clearly not about machine learning or data, such as geography, cooking, sports or celebrities"}},
              "default": "retrieve"},
    "grade": {"q": {"type": "noul", "instructions": "Does the passage contain information needed to answer the question?"},
              "default": "no"},
    "grounded": {"q": {"type": "noul", "instructions": "Is every claim in the answer supported by the passages?"},
                 "default": "no"},
    "sufficient": {"q": {"type": "noul", "instructions": "Does the answer directly address the question?"},
                   "default": "no"},
}


def labels_of(gate):
    q = GATES[gate]["q"]
    return list(q["criteria"]) if q["type"] == "choice" else ["yes", "no"]


class Gate:
    last_s = 0.0   # non-LLM compute seconds of the latest call (LLM time is tracked by LLM.total_s)
    llm_used = 0   # decisions answered by an LLM

    def decide(self, gate, text):
        return self.decide_many(gate, [text])[0]


class LLMGate(Gate):
    def __init__(self, llm, model=GEMMA):
        self.llm, self.model, self.unparsed = llm, model, 0

    # ponytail: one LLM call per text, sequential, like typical LangGraph tutorials; a parallel judge would be faster
    def decide_many(self, gate, texts):
        q, labels, out = GATES[gate]["q"], labels_of(gate), []
        opts = "\n".join(f"- {k}: {v}" for k, v in q.get("criteria", {}).items())
        for t in texts:
            prompt = (f"{q['instructions']}\n{opts}\n\nInput:\n{t}\n\n"
                      f"Reply with exactly one word from: {', '.join(labels)}.")
            words = re.findall(r"[a-z_]+", self.llm.chat(self.model, prompt, max_tokens=16).lower())
            hit = next((w for w in words if w in labels), None)
            if hit is None:
                self.unparsed += 1
            out.append((hit or GATES[gate]["default"], 1.0 if hit else 0.0))
        self.llm_used += len(texts)
        self.last_s = 0.0
        return out


class SkGate(Gate):
    """TF-IDF + logistic regression: the 'does a trivial model already do it?' baseline."""
    def __init__(self):
        self.m = {}

    def fit(self, gate, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.m[gate] = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                     LogisticRegression(max_iter=1000)).fit(texts, labels)
        return self

    def decide_many(self, gate, texts):
        t0 = time.perf_counter()
        m = self.m[gate]
        out = [(str(m.classes_[p.argmax()]), float(p.max())) for p in m.predict_proba(texts)]
        self.last_s = time.perf_counter() - t0
        return out


class LayaGate(Gate):
    def __init__(self, agents):
        self.agents = agents  # {gate: laya agent}

    @classmethod
    def load(cls, ckpts):
        """ckpts = {gate: (checkpoint_dir, calibration_json_path_or_None)}"""
        for gate, (ck, _) in ckpts.items():
            if not os.path.isfile(os.path.join(ck, "model.safetensors")):  # laya.load would take a missing folder for a Hugging Face repo and fail with a 401
                raise FileNotFoundError(f"fine-tuned weights '{ck}/model.safetensors' for gate '{gate}' are missing: attach notebook 04's "
                                        "OUTPUT as an input (it must have been saved with Save & Run All), or re-run notebook 04")
        import laya
        agents = {}
        for gate, (ck, cal) in ckpts.items():
            a = laya.load(ck)
            if cal:
                a.load_calibration(cal)
            agents[gate] = a
        return cls(agents)

    def decide_many(self, gate, texts):
        t0 = time.perf_counter()
        res = self.agents[gate].predict_batch([{"body": t} for t in texts], {gate: GATES[gate]["q"]}, batch_size=64)
        out = []
        for r in res:
            a = r["answers"][gate]
            if GATES[gate]["q"]["type"] == "noul":
                p = a["noul"]
                # Binning (calibration) is applied to answer_confidence only; use it for confidence
                conf = a["answer_confidence"]
                out.append(("yes", conf) if p >= 0.5 else ("no", conf))
            else:
                out.append((a["choice"], a["answer_confidence"]))
        self.last_s = time.perf_counter() - t0
        return out


class Cascade(Gate):
    """Ask `fast`; if its confidence is below tau, ask `slow` instead."""
    def __init__(self, fast, slow, tau):
        self.fast, self.slow, self.tau = fast, slow, tau
        self.escalated = self.seen = 0

    def decide_many(self, gate, texts):
        out = self.fast.decide_many(gate, texts)
        s = self.fast.last_s
        for i, (_, conf) in enumerate(out):
            if conf < self.tau:
                out[i] = self.slow.decide_many(gate, [texts[i]])[0]
                s += self.slow.last_s
                self.escalated += 1
        self.seen += len(texts)
        self.last_s = s
        return out

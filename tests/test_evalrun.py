from gatekeep.evalrun import agreement, judge, make_app, run_variant, summarize
from gatekeep.gates import Gate, LLMGate
from gatekeep.graph import REFUSAL


class FakeLLM:
    def __init__(self, reply="CORRECT"):
        self.reply, self.logical_calls, self.total_s = reply, 0, 0.0

    def chat(self, model, prompt, **kw):
        self.logical_calls += 1
        self.total_s += 0.5
        return self.reply


class StubApp:
    def __init__(self, llm, answer="an answer", gate_s=0.1):
        self.llm, self.answer, self.gate_s = llm, answer, gate_s

    def invoke(self, s):
        self.llm.chat("m", "generate")
        return {"answer": self.answer, "docs": [{"text": "t"}], "gate_calls": 3, "gate_s": self.gate_s, "retr_s": 0.2}


ITEM = {"id": "t-0", "q": "q?", "a": "ref", "answerable": True}


def test_run_variant_counts_calls_and_latency_before_judging():
    llm = FakeLLM("CORRECT")
    rows = run_variant(StubApp(llm), LLMGate(llm), llm, [ITEM])
    r = rows[0]
    assert r["llm_calls"] == 1 and r["correct"] == 1
    assert abs(r["latency_s"] - (0.5 + 0.1 + 0.2)) < 1e-9  # grading calls are not charged to the system


def test_judge_unanswerable_rewards_refusal():
    j = judge(FakeLLM("CORRECT"), {"q": "x", "a": None, "answerable": False}, {"answer": REFUSAL})
    assert j["correct"] == 1 and j["faithful"] is None


def test_judge_incorrect_is_zero():
    assert judge(FakeLLM("INCORRECT"), ITEM, {"answer": "wrong", "docs": []})["correct"] == 0


def test_summarize_means_and_ci():
    s = summarize([{"correct": 1, "faithful": 1, "llm_calls": 4, "gate_llm_calls": 0, "latency_s": 1.0},
                   {"correct": 0, "faithful": None, "llm_calls": 6, "gate_llm_calls": 2, "latency_s": 3.0}])
    assert s["correct"][0] == 0.5 and s["llm_calls"][0] == 5 and s["faithful"][0] == 1.0
    assert s["correct"][1] <= 0.5 <= s["correct"][2]


def test_make_app_builds_every_variant():
    class G(Gate):
        def decide_many(self, g, t):
            return [("yes", 1.0)] * len(t)

    class Idx:
        def search(self, q, k=20):
            return [{"id": 0, "text": "t"}]

        def rerank(self, q, h, k=5):
            return h

    llm = FakeLLM("an answer")
    for v in ("V0", "V1", "V2", "V3"):
        app, llm_gate = make_app(v, Idx(), llm, laya=G())
        assert hasattr(app, "invoke") and llm_gate is not None


def test_agreement_is_the_share_of_matching_verdicts():
    class Split(FakeLLM):
        def __init__(self, ultra_reply):
            super().__init__()
            self.ultra_reply = ultra_reply

        def chat(self, model, prompt, **kw):
            return self.ultra_reply if model == "ultra" else "CORRECT"

    items, rows = [ITEM], [{"id": "t-0", "answer": "x"}]
    assert agreement(Split("CORRECT"), items, rows, n=1) == 1.0
    assert agreement(Split("INCORRECT"), items, rows, n=1) == 0.0

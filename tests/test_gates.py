from gatekeep.gates import Cascade, Gate, LLMGate, SkGate


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, model, prompt, **kw):
        return self.reply


class Fixed(Gate):
    def __init__(self, label, conf):
        self.label, self.conf = label, conf

    def decide_many(self, gate, texts):
        return [(self.label, self.conf)] * len(texts)


def test_llm_gate_parses_yes_no():
    assert LLMGate(FakeLLM("Yes, it does.")).decide("grade", "x") == ("yes", 1.0)
    assert LLMGate(FakeLLM("no")).decide("grade", "x") == ("no", 1.0)


def test_llm_gate_parses_choice_label():
    assert LLMGate(FakeLLM("off_topic")).decide("route", "x")[0] == "off_topic"


def test_llm_gate_unparseable_uses_default_and_counts():
    g = LLMGate(FakeLLM("???"))
    assert g.decide("route", "x") == ("retrieve", 0.0)
    assert g.decide("grade", "x") == ("no", 0.0)
    assert g.unparsed == 2


def test_llm_gate_counts_every_decision_it_answers():
    g = LLMGate(FakeLLM("yes"))
    g.decide_many("grade", ["a", "b", "c"])
    assert g.llm_used == 3


def test_cascade_escalates_only_below_tau():
    slow = LLMGate(FakeLLM("no"))
    c = Cascade(Fixed("yes", 0.6), slow, tau=0.8)
    assert c.decide("grade", "x") == ("no", 1.0) and c.escalated == 1
    c2 = Cascade(Fixed("yes", 0.9), slow, tau=0.8)
    assert c2.decide("grade", "x") == ("yes", 0.9) and c2.escalated == 0 and c2.seen == 1


def test_sk_gate_learns_a_trivial_rule():
    sk = SkGate()
    texts = ["good great fine", "good nice great", "bad awful poor", "bad poor awful"] * 3
    sk.fit("grade", texts, ["yes", "yes", "no", "no"] * 3)
    (l1, p1), (l2, _) = sk.decide_many("grade", ["great good", "awful bad"])
    assert (l1, l2) == ("yes", "no") and p1 > 0.5


def test_laya_gate_load_names_the_missing_folder_instead_of_asking_the_hub(tmp_path):
    import pytest
    from gatekeep.gates import LayaGate
    with pytest.raises(FileNotFoundError, match="notebook 04"):
        LayaGate.load({"route": (str(tmp_path / "models" / "route"), None)})


def test_laya_gate_load_rejects_a_folder_without_weights(tmp_path):
    import pytest
    from gatekeep.gates import LayaGate
    (tmp_path / "route").mkdir()  # the folder exists (e.g. a restored ckpts.json) but the 3 GB of weights did not travel
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        LayaGate.load({"route": (str(tmp_path / "route"), None)})


class FakeLaya:
    """Fake Laya agent for testing; predict_batch returns fixed results."""
    def __init__(self, results):
        self.results = results  # list of dicts with the structure predict_batch returns

    def predict_batch(self, states, questions, batch_size=64):
        return self.results


def test_laya_gate_noul_takes_calibrated_answer_confidence():
    """When answer_confidence is binned (calibrated), noul gates should use it for confidence, not the raw noul value."""
    from gatekeep.gates import LayaGate
    fake = FakeLaya([{"answers": {"grade": {"noul": 0.8, "answer_confidence": 0.62}}}])
    gate = LayaGate({"grade": fake})
    assert gate.decide_many("grade", ["x"]) == [("yes", 0.62)]


def test_laya_gate_noul_label_from_raw_noul_confidence_from_binned():
    """Confirm the label comes from raw noul but confidence from answer_confidence."""
    from gatekeep.gates import LayaGate
    fake = FakeLaya([{"answers": {"grade": {"noul": 0.3, "answer_confidence": 0.55}}}])
    gate = LayaGate({"grade": fake})
    assert gate.decide_many("grade", ["x"]) == [("no", 0.55)]

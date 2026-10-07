from gatekeep.demo import ask
from gatekeep.graph import REFUSAL
from test_graph import FakeIndex, FakeLLM, Scripted

TAUS = {g: 0.8 for g in ("route", "grade", "grounded", "sufficient")}


def laya(conf, route="retrieve"):
    return Scripted(lambda g, t: (route, conf) if g == "route" else ("yes", conf))


def test_trace_names_each_step_and_who_decided():
    answer, trace, passages, secs = ask("what is bagging?", FakeIndex(), FakeLLM(), laya(1.0), TAUS)
    text = "\n".join(trace)
    assert answer == "an answer" and len(passages) == 5 and secs >= 0
    assert "route -> retrieve" in text and "grade: kept 5 passages" in text and "(Laya," in text
    assert "Gemma" not in text


def test_low_confidence_is_shown_as_handed_to_gemma():
    _, trace, _, _ = ask("what is bagging?", FakeIndex(), FakeLLM(), laya(0.1), TAUS)
    assert "Laya, then Gemma for 1 of 1" in "\n".join(trace)


def test_off_topic_shows_a_refusal_and_no_passages():
    answer, trace, passages, _ = ask("capital of France?", FakeIndex(), FakeLLM(), laya(1.0, "off_topic"), TAUS)
    assert "user guide" in answer and answer != REFUSAL and passages == [] and trace[0] == "route -> off_topic"


from gatekeep.demo import ask_events


class CountingLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.gate_prompts = 0

    def chat(self, model, prompt, **kw):
        if prompt.startswith(("Does this", "Does the", "Is every", "Is this")):
            self.gate_prompts += 1  # the LLM gate's questions
        return super().chat(model, prompt, **kw)


def test_events_come_in_order_and_end_with_answer_then_done():
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), laya(1.0), TAUS, "careful"))
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "step" and evs[0]["node"] == "route"
    assert kinds[-3:] == ["passages", "answer", "done"]
    assert evs[-2] == {"type": "answer", "text": "an answer", "refused": False, "reason": None}
    assert len(evs[-3]["items"]) == 5 and set(evs[-3]["items"][0]) == {"text", "section", "page"}
    gates = [e for e in evs if e["type"] == "gate"]
    assert {g["gate"] for g in gates} == {"route", "grade", "grounded", "sufficient"} and all(g["who"] == "Laya" for g in gates)


def test_fast_mode_never_asks_the_llm_to_judge_even_when_laya_is_unsure():
    llm = CountingLLM()
    evs = list(ask_events("what is bagging?", FakeIndex(), llm, laya(0.1), TAUS, "fast"))
    assert llm.gate_prompts == 0 and all(e["who"] == "Laya" for e in evs if e["type"] == "gate")


def test_careful_mode_hands_unsure_decisions_to_gemma():
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), laya(0.1), TAUS, "careful"))
    assert any(e["type"] == "gate" and e["who"] == "Laya, then Gemma for 1 of 1" for e in evs)


def test_refusals_carry_their_reason():
    off = list(ask_events("capital of France?", FakeIndex(), FakeLLM(), laya(1.0, "off_topic"), TAUS, "fast"))
    assert off[-2] == {"type": "answer", "text": REFUSAL, "refused": True, "reason": "off_topic"}
    nothing = Scripted(lambda g, t: ("retrieve", 1.0) if g == "route" else ("no", 1.0))
    none = list(ask_events("x", FakeIndex(), FakeLLM(), nothing, TAUS, "fast"))
    assert none[-2]["reason"] == "no_passage" and none[-3]["items"] == []


def test_unknown_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        list(ask_events("x", FakeIndex(), FakeLLM(), laya(1.0), TAUS, "turbo"))

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
    assert evs[-2] == {"type": "answer", "text": "an answer", "refused": False, "flagged": False, "reason": None}
    assert len(evs[-3]["items"]) == 5 and set(evs[-3]["items"][0]) == {"text", "section", "page", "source"}
    gates = [e for e in evs if e["type"] == "gate"]
    assert {g["gate"] for g in gates} == {"route", "grade", "grounded", "sufficient"} and all(g["who"] == "Laya" for g in gates)


def test_fast_mode_never_asks_the_llm_to_judge_when_laya_keeps_a_passage():
    llm = CountingLLM()
    evs = list(ask_events("what is bagging?", FakeIndex(), llm, laya(0.1), TAUS, "fast"))
    assert llm.gate_prompts == 0 and all(e["who"] == "Laya" for e in evs if e["type"] == "gate")


def test_careful_mode_hands_unsure_decisions_to_gemma():
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), laya(0.1), TAUS, "careful"))
    assert any(e["type"] == "gate" and e["who"] == "Laya, then Gemma for 1 of 1" for e in evs)


def test_refusals_carry_their_reason():
    off = list(ask_events("capital of France?", FakeIndex(), FakeLLM(), laya(1.0, "off_topic"), TAUS, "fast"))
    assert off[-2] == {"type": "answer", "text": REFUSAL, "refused": True, "flagged": False, "reason": "off_topic"}
    nothing = Scripted(lambda g, t: ("retrieve", 1.0) if g == "route" else ("no", 1.0))
    none = list(ask_events("x", FakeIndex(), FakeLLM(), nothing, TAUS, "fast"))
    assert none[-2]["reason"] == "no_passage" and none[-3]["items"] == []


def test_unknown_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        list(ask_events("x", FakeIndex(), FakeLLM(), laya(1.0), TAUS, "turbo"))


class YesGradeLLM(FakeLLM):
    def chat(self, model, prompt, **kw):
        return "yes" if prompt.startswith("Does the passage") else super().chat(model, prompt, **kw)


def test_fast_mode_asks_gemma_only_when_laya_rejects_every_passage():
    strict = Scripted(lambda g, t: ("retrieve", 1.0) if g == "route" else ("no", 0.9) if g == "grade" else ("yes", 0.9))
    evs = list(ask_events("what is the difference between bagging and boosting?", FakeIndex(), YesGradeLLM(), strict, TAUS, "fast"))
    grade = [e for e in evs if e["type"] == "gate" and e["gate"] == "grade"]
    assert grade[0]["who"] == "Laya, then Gemma for 5 of 5" and grade[0]["labels"] == ["yes"] * 5
    assert evs[-2]["refused"] is False
    others = [e for e in evs if e["type"] == "gate" and e["gate"] != "grade"]
    assert all(e["who"] == "Laya" for e in others)


def test_fast_mode_recheck_asks_gemma_the_lenient_question_about_compound_questions():
    strict = Scripted(lambda g, t: ("retrieve", 1.0) if g == "route" else ("no", 0.9) if g == "grade" else ("yes", 0.9))
    class Spy(YesGradeLLM):
        def chat(self, model, prompt, **kw):
            self.prompts.append(prompt)
            return super().chat(model, prompt, **kw)
    llm = Spy()
    list(ask_events("what is bagging and boosting?", FakeIndex(), llm, strict, TAUS, "fast"))
    assert any(p.startswith("Does the passage help answer any part") for p in llm.prompts)



def counting(fn):
    """A Laya stand-in whose answer depends on how many texts it has judged for that gate so far."""
    seen = {}

    def decide(gate, text):
        seen[gate] = seen.get(gate, 0) + 1
        return fn(gate, seen[gate])
    return Scripted(decide)


def escalations(evs):
    return [e for e in evs if e["type"] == "step" and e["node"] == "escalate"]


def test_auto_stays_on_the_fast_path_when_it_finds_an_answer():
    llm = CountingLLM()
    evs = list(ask_events("what is bagging?", FakeIndex(), llm, laya(1.0), TAUS, "auto"))
    assert escalations(evs) == [] and llm.gate_prompts == 0
    assert [e["type"] for e in evs][-3:] == ["passages", "answer", "done"] and evs[-2]["refused"] is False


def test_auto_tries_careful_when_fast_finds_no_passage():
    # fast: Laya rejects all five and Gemma's re-check parses to "no"; careful: Laya now keeps them
    nope_then_yes = counting(lambda g, n: ("retrieve", 1.0) if g == "route" else (("no" if n <= 5 else "yes"), 1.0))
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), nope_then_yes, TAUS, "auto"))
    assert len(escalations(evs)) == 1
    answer = [e for e in evs if e["type"] == "answer"]
    assert len(answer) == 1 and answer[0]["refused"] is False  # the failed fast attempt shows no answer of its own
    assert evs[-1]["type"] == "done" and len([e for e in evs if e["type"] == "done"]) == 1


def test_auto_tries_careful_when_fast_ends_with_a_flagged_answer():
    bad_then_good = counting(lambda g, n: ("retrieve", 1.0) if g == "route" else
                             ("no" if g == "grounded" and n <= 3 else "yes", 1.0))
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), bad_then_good, TAUS, "auto"))
    assert len(escalations(evs)) == 1 and evs[-2]["flagged"] is False


def test_auto_does_not_escalate_an_off_topic_refusal():
    evs = list(ask_events("capital of France?", FakeIndex(), FakeLLM(), laya(1.0, "off_topic"), TAUS, "auto"))
    assert escalations(evs) == [] and evs[-2]["reason"] == "off_topic"


def test_careful_mode_asks_gemma_the_lenient_grade_question():
    class Spy(YesGradeLLM):
        def chat(self, model, prompt, **kw):
            self.prompts.append(prompt)
            return super().chat(model, prompt, **kw)
    llm = Spy()
    list(ask_events("what is bagging and boosting?", FakeIndex(), llm, laya(0.1), TAUS, "careful"))
    assert any(p.startswith("Does the passage help answer any part") for p in llm.prompts)

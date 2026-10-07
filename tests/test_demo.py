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

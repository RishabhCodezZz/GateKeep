import pytest
from gatekeep.llm import LLM


class Fake:
    """Mimics openai.OpenAI().chat.completions.create."""
    def __init__(self):
        self.n = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.n += 1
        msg = type("M", (), {"content": "hi"})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def test_second_identical_call_is_cached(tmp_path):
    f = Fake()
    llm = LLM(str(tmp_path / "c.sqlite"), client=f)
    assert llm.chat("m", "q") == "hi"
    assert llm.chat("m", "q") == "hi"
    assert f.n == 1 and llm.logical_calls == 2


def test_cached_call_replays_original_latency(tmp_path):
    llm = LLM(str(tmp_path / "c.sqlite"), client=Fake())
    llm.chat("m", "q")
    first = llm.total_s
    llm.chat("m", "q")
    assert llm.total_s == pytest.approx(2 * first, rel=1e-9)


def test_different_temperature_is_a_different_call(tmp_path):
    f = Fake()
    llm = LLM(str(tmp_path / "c.sqlite"), client=f)
    llm.chat("m", "q", temperature=0.0)
    llm.chat("m", "q", temperature=0.7)
    assert f.n == 2

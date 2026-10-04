import pytest
from gatekeep.llm import GEMMA, LLM, SUPER, ULTRA


class Fake:
    """Mimics openai.OpenAI().chat.completions.create."""
    def __init__(self):
        self.n = 0
        self.last = {}
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.n += 1
        self.last = kw
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


def test_alias_resolves_to_each_backends_model_id(tmp_path):
    for backend, gemma, ultra, sup in (("nim", "google/gemma-4-31b-it", "nvidia/nemotron-3-ultra-550b-a55b",
                                        "nvidia/nemotron-3-super-120b-a12b"),
                                       ("ollama", "gemma4:31b", "nemotron-3-ultra", "nemotron-3-super")):
        f = Fake()
        llm = LLM(str(tmp_path / f"{backend}.sqlite"), client=f, backend=backend)
        llm.chat(GEMMA, "q")
        assert f.last["model"] == gemma
        llm.chat(ULTRA, "q")
        assert f.last["model"] == ultra
        llm.chat(SUPER, "q")
        assert f.last["model"] == sup


def test_same_prompt_on_another_backend_is_not_a_cache_hit(tmp_path):
    f, db = Fake(), str(tmp_path / "shared.sqlite")
    LLM(db, client=f, backend="nim").chat(GEMMA, "q")
    LLM(db, client=f, backend="ollama").chat(GEMMA, "q")
    assert f.n == 2


def test_unknown_backend_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        LLM(str(tmp_path / "c.sqlite"), client=Fake(), backend="nope")

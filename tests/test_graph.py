from gatekeep.gates import Gate
from gatekeep.graph import REFUSAL, build_agents_graph, build_graph, clean_query


class FakeIndex:
    def search(self, q, k=20):
        return [{"id": i, "text": f"t{i}"} for i in range(5)]

    def rerank(self, q, hits, k=5):
        return hits[:k]


class FakeLLM:
    def __init__(self):
        self.logical_calls, self.total_s, self.prompts = 0, 0.0, []

    def chat(self, model, prompt, **kw):
        self.logical_calls += 1
        self.prompts.append(prompt)
        return "short query" if prompt.startswith("Rewrite") else "an answer"


class NewDocsIndex:
    """A different set of passages for every query, like a rewrite that finds something new."""
    def search(self, q, k=20):
        return [{"id": i, "text": f"{q}-t{i}"} for i in range(5)]

    def rerank(self, q, hits, k=5):
        return hits[:k]


class Scripted(Gate):
    def __init__(self, fn):
        self.fn = fn

    def decide_many(self, gate, texts):
        return [self.fn(gate, t) for t in texts]


def gates(fn):
    return {g: Scripted(fn) for g in ("route", "grade", "grounded", "sufficient")}


def app(fn, llm=None, **kw):
    return build_graph(FakeIndex(), llm or FakeLLM(), gates(fn), **kw)


def happy(gate, text):
    return ("retrieve", 1.0) if gate == "route" else ("yes", 1.0)


def test_off_topic_refuses_without_retrieval():
    r = app(lambda g, t: ("off_topic", 1.0) if g == "route" else ("yes", 1.0)).invoke({"q": "x"})
    assert r["answer"] == REFUSAL and "docs" not in r


def no(g, t):
    return ("retrieve", 1.0) if g == "route" else ("no", 1.0)


def test_no_relevant_docs_rewrites_twice_then_refuses():
    r = build_graph(NewDocsIndex(), FakeLLM(), gates(no)).invoke({"q": "x"})
    assert r["answer"] == REFUSAL and r["rewrites"] == 2


def test_a_rewrite_that_finds_the_same_passages_stops_without_grading_them_again():
    r = app(no).invoke({"q": "x"})
    assert r["answer"] == REFUSAL and r["rewrites"] == 1 and r["gate_calls"] == 1 + 5  # route + one grading of 5


def test_agents_graph_also_stops_on_repeated_passages():
    r = build_agents_graph(FakeIndex(), FakeLLM(), gates(no)).invoke({"q": "x"})
    assert r["answer"] == REFUSAL and r["rewrites"] == 1 and r["gate_calls"] == 1 + 5


def test_clean_query_drops_markdown_quotes_and_chatter():
    assert clean_query('**"bagging vs boosting"**', "q") == "bagging vs boosting"
    assert clean_query("Sure, try:\n\n**boosting**\n- other idea", "q") == "boosting"
    assert clean_query("\n`boosting`\nOther ideas:\n- x", "q") == "boosting"
    assert clean_query("**", "what is x?") == "what is x?"
    assert clean_query("a b c d e f g h i j k l m", "q") == "q"  # a reply that is not a short query falls back to the question


def test_the_rewritten_query_is_cleaned_before_searching():
    class Markdown(FakeLLM):
        def chat(self, model, prompt, **kw):
            return "**\"bagging vs boosting\"**" if prompt.startswith("Rewrite") else "an answer"
    r = build_graph(NewDocsIndex(), Markdown(), gates(no)).invoke({"q": "x"})
    assert r["query"] == "bagging vs boosting"


def test_failed_check_regenerates_once_then_passes():
    seen = {"n": 0}

    def fn(g, t):
        if g == "route":
            return ("retrieve", 1.0)
        if g == "grounded":
            seen["n"] += 1
            return ("no", 1.0) if seen["n"] == 1 else ("yes", 1.0)
        return ("yes", 1.0)

    r = app(fn).invoke({"q": "x"})
    assert r["regens"] == 1 and r["answer"] == "an answer" and not r.get("flagged")


def test_gives_up_and_flags_after_max_regen():
    r = app(lambda g, t: ("retrieve", 1.0) if g == "route" else ("yes", 1.0) if g == "grade" else ("no", 1.0)).invoke({"q": "x"})
    assert r["flagged"] and r["regens"] == 3


def test_baseline_without_gates_makes_no_gate_calls():
    r = build_graph(FakeIndex(), FakeLLM(), {}, use_gates=False).invoke({"q": "x"})
    assert r["answer"] == "an answer" and "gate_calls" not in r


def test_gate_calls_are_counted():
    r = app(happy).invoke({"q": "x"})
    assert r["gate_calls"] == 1 + 5 + 1 + 1  # route + 5 chunks + grounded + sufficient



def test_agents_graph_passes_critic_feedback_to_writer():
    llm = FakeLLM()
    seen = {"n": 0}

    def fn(g, t):
        if g == "route":
            return ("retrieve", 1.0)
        if g == "grounded":
            seen["n"] += 1
            return ("no", 1.0) if seen["n"] == 1 else ("yes", 1.0)
        return ("yes", 1.0)

    r = build_agents_graph(FakeIndex(), llm, gates(fn)).invoke({"q": "x"})
    assert r["regens"] == 1 and r["answer"] == "an answer"
    assert any("not supported by the passages" in p for p in llm.prompts)


def test_flat_graph_sends_no_feedback_to_the_writer():
    llm = FakeLLM()
    seen = {"n": 0}

    def fn(g, t):
        if g == "route":
            return ("retrieve", 1.0)
        if g == "grounded":
            seen["n"] += 1
            return ("no", 1.0) if seen["n"] == 1 else ("yes", 1.0)
        return ("yes", 1.0)

    app(fn, llm).invoke({"q": "x"})
    assert not any("reviewer said" in p for p in llm.prompts)


def test_agents_graph_refuses_when_nothing_relevant():
    r = build_agents_graph(FakeIndex(), FakeLLM(),
                           gates(lambda g, t: ("retrieve", 1.0) if g == "route" else ("no", 1.0))).invoke({"q": "x"})
    assert r["answer"] == REFUSAL


def test_evidence_uses_the_fine_tuned_window():
    from gatekeep.graph import EVIDENCE_CHARS, evidence
    assert EVIDENCE_CHARS == 2000
    docs = [{"text": "a" * 900}, {"text": "b" * 900}, {"text": "c" * 900}]
    assert len(evidence(docs)) == 900 + 2 + 900  # two whole passages fit, the third would pass 2,000

from gatekeep import data

ITEMS = [{"id": "a", "q": "What is X?", "a": "X is Y.", "gold": [0], "chapter": "c1", "answerable": True},
         {"id": "b", "q": "What is LoRA?", "a": None, "gold": [], "chapter": "none", "answerable": False}]
BY_ID = {0: {"id": 0, "text": "X is Y."}, 1: {"id": 1, "text": "Other text."}}


class FakeIndex:
    def search(self, q, k=20):
        return [BY_ID[0], BY_ID[1]]

    def rerank(self, q, hits, k=5):
        return hits[:k]


class FakeLLM:
    def chat(self, model, prompt, **kw):
        return "YES" if "Reply YES or NO" in prompt else "an altered answer"


def test_laya_row_encodes_choice_and_noul():
    c = data.laya_row("route", "hi", "direct")
    assert c["answers"]["route"] == {"choice": "direct"} and c["state"] == {"body": "hi"}
    n = data.laya_row("grade", "t", "no")
    assert n["answers"]["grade"] == {"noul": False}
    assert n["questions"]["grade"]["type"] == "noul"


def test_g1_labels_by_answerability():
    rows = data.g1_rows(ITEMS, ["hello there"])
    assert {r["label"] for r in rows} == {"retrieve", "out_of_scope", "direct"}


def test_g2_gold_positive_and_relabeled_negative():
    rows = data.g2_rows(FakeLLM(), FakeIndex(), ITEMS, BY_ID)
    assert rows[0]["label"] == "yes"
    assert any("Other text." in r["text"] for r in rows)  # a retrieved non-gold chunk was judged


class Grader(FakeLLM):
    def __init__(self, verdict):
        self.verdict = verdict

    def chat(self, model, prompt, **kw):
        return self.verdict if "Reply YES or NO" in prompt else "Gemma's answer"


def test_g3_rows_take_their_label_from_the_grader():
    yes = data.g3_rows(Grader("YES"), FakeIndex(), ITEMS, BY_ID)
    no = data.g3_rows(Grader("NO"), FakeIndex(), ITEMS, BY_ID)
    assert [r["label"] for r in yes] == ["yes"] and [r["label"] for r in no] == ["no"]
    assert "Gemma's answer" in yes[0]["text"]  # the row holds a real answer, never a corrupted one


def test_g3_rows_can_sample_more_than_one_answer_per_question():
    rows = data.g3_rows(Grader("YES"), FakeIndex(), ITEMS, BY_ID, temps=(0.0, 0.7))
    assert len(rows) == 2


class Answerer(FakeLLM):
    """Answers with a marker built from the question, so we can tell whose answer a row contains."""
    def chat(self, model, prompt, **kw):
        q = prompt.rsplit("Question: ", 1)[1].split("\n")[0]
        return f"ANSWER-TO[{q}]"


def test_g4_negative_is_another_questions_answer_in_the_same_style():
    items = [{"id": f"i{k}", "q": f"Q{k}?", "a": f"ref{k}", "gold": [0], "chapter": "c1", "answerable": True} for k in range(3)]
    rows = data.g4_rows(Answerer(), items, BY_ID)
    assert len(rows) == 6 and {r["label"] for r in rows} == {"yes", "no"}
    for yes_row, no_row in zip(rows[0::2], rows[1::2]):
        question = yes_row["text"].split("\n")[0]
        assert no_row["text"].split("\n")[0] == question                      # same question
        assert f"ANSWER-TO[{question.removeprefix('Question: ')}]" in yes_row["text"]    # positive: its own answer
        assert f"ANSWER-TO[{question.removeprefix('Question: ')}]" not in no_row["text"]  # negative: someone else's
        assert no_row["text"].split("Answer: ")[1].startswith("ANSWER-TO[")   # both are Gemma-written: no style giveaway


def test_g4_skips_questions_with_no_partner_and_unanswerable_ones():
    assert data.g4_rows(Answerer(), ITEMS, BY_ID) == []  # only one answerable item: nothing to pair it with


def test_write_gate_data_writes_plain_and_laya_files(tmp_path):
    data.write_gate_data({"grade": [{"text": "t", "label": "yes"}]}, "train", str(tmp_path))
    assert (tmp_path / "grade_train.jsonl").read_text().strip().startswith('{"text"')
    assert '"noul": true' in (tmp_path / "grade_train.laya.jsonl").read_text()

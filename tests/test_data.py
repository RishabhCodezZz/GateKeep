from gatekeep import data

ITEMS = [{"id": "a", "q": "What is X?", "a": "X is Y.", "gold": [0], "answerable": True},
         {"id": "b", "q": "What is LoRA?", "a": None, "gold": [], "answerable": False}]
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


def test_g3_synthetic_pairs_have_both_labels():
    rows = data.g3_rows(FakeLLM(), FakeIndex(), ITEMS, BY_ID)
    assert {r["label"] for r in rows} == {"yes", "no"}


def test_g3_real_rows_take_their_label_from_the_grader():
    class Grader(FakeLLM):
        def __init__(self, verdict):
            self.verdict = verdict

        def chat(self, model, prompt, **kw):
            return self.verdict if "Reply YES or NO" in prompt else "Gemma's answer"

    yes = data.g3_rows(Grader("YES"), FakeIndex(), ITEMS, BY_ID, real=True)
    no = data.g3_rows(Grader("NO"), FakeIndex(), ITEMS, BY_ID, real=True)
    assert [r["label"] for r in yes] == ["yes"] and [r["label"] for r in no] == ["no"]
    assert "Gemma's answer" in yes[0]["text"]  # the row holds the real answer, not a corrupted one


def test_g4_pairs_have_both_labels():
    assert {r["label"] for r in data.g4_rows(FakeLLM(), ITEMS)} == {"yes", "no"}


def test_write_gate_data_writes_plain_and_laya_files(tmp_path):
    data.write_gate_data({"grade": [{"text": "t", "label": "yes"}]}, "train", str(tmp_path))
    assert (tmp_path / "grade_train.jsonl").read_text().strip().startswith('{"text"')
    assert '"noul": true' in (tmp_path / "grade_train.laya.jsonl").read_text()

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


def test_gold_for_encodes_the_probabilities_laya_trains_on():
    assert data.gold_for("grade", "yes") == {"probabilities": {"true": 1.0, "false": 0.0}}
    assert data.gold_for("grade", "no") == {"probabilities": {"true": 0.0, "false": 1.0}}
    assert data.gold_for("route", "direct") == {"probabilities": {"retrieve": 0.0, "direct": 1.0, "off_topic": 0.0}}


def test_balance_rows_repeats_minority_classes_up_to_a_share_of_the_majority():
    from collections import Counter
    rows = [{"text": str(i), "label": "a"} for i in range(100)] + [{"text": "x", "label": "b"}] * 7
    counts = Counter(r["label"] for r in data.balance_rows(rows, min_share=0.25))
    assert counts["a"] == 100 and counts["b"] == 25


def test_balance_rows_leaves_a_balanced_set_alone():
    rows = [{"text": "1", "label": "a"}, {"text": "2", "label": "b"}]
    assert data.balance_rows(rows) == rows


def test_g1_sends_every_book_question_to_retrieval_and_adds_the_new_classes():
    rows = data.g1_rows(ITEMS, ["hello there"], ["what is a GAN?"], ["capital of France?"])
    labels = {r["text"]: r["label"] for r in rows}
    assert all(labels[it["q"]] == "retrieve" for it in ITEMS)  # answerable or not: grading decides coverage
    assert labels["what is a GAN?"] == "retrieve" and labels["capital of France?"] == "off_topic"
    assert labels["hello there"] == "direct"


def test_split_messages_shuffles_before_cutting_and_keeps_every_message_once():
    msgs = [f"m{i}" for i in range(100)]
    sp = data.split_messages(msgs)
    assert [len(sp[k]) for k in ("train", "dev", "test")] == [60, 15, 25]
    assert sorted(sp["train"] + sp["dev"] + sp["test"]) == sorted(msgs)
    assert sp["train"] != msgs[:60]  # shuffled, so one kind of message cannot fill a whole split


def test_gen_ml_short_and_gen_off_topic_use_their_own_rules():
    class Spy(FakeLLM):
        prompts = []

        def chat(self, model, prompt, **kw):
            self.prompts.append(prompt)
            return "a message"

    llm = Spy()
    data.gen_ml_short(llm, 5)
    data.gen_off_topic(llm, 5)
    assert any("about machine learning" in p and "must be short" in p for p in llm.prompts)
    assert any("must not be about machine learning, data or programming" in p for p in llm.prompts)


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


def test_write_gate_data_writes_one_plain_file_per_gate_and_split(tmp_path):
    data.write_gate_data({"grade": [{"text": "t", "label": "yes"}]}, "train", str(tmp_path))
    assert (tmp_path / "grade_train.jsonl").read_text().strip().startswith('{"text"')
    assert [p.name for p in tmp_path.iterdir()] == ["grade_train.jsonl"]


class Lines(FakeLLM):
    """Replies with a messy numbered list, the way a model really answers 'one per line'."""
    def chat(self, model, prompt, **kw):
        return '1. Hello!\n2. "Thanks a lot"\n- bye\n\n* \n3) What is 2+2?\n'


def test_gen_direct_reads_one_message_per_line_and_cleans_numbering_and_quotes():
    out = data.gen_direct(Lines(), n=100)
    assert out == ["Hello!", "Thanks a lot", "bye", "What is 2+2?"]  # de-duplicated across the topic batches, blanks dropped


def test_gen_direct_never_returns_more_than_n():
    class Many(FakeLLM):
        def chat(self, model, prompt, **kw):
            return "\n".join(f"message {i}" for i in range(50))

    assert len(data.gen_direct(Many(), n=10)) == 10

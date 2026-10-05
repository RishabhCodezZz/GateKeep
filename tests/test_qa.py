from gatekeep import qa


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, model, prompt, **kw):
        return self.reply


CH = [{"id": i, "text": f"text about topic {i}", "chapter": f"Ch{i % 4}", "section": ""} for i in range(8)]


def test_parse_json_handles_fences_and_noise():
    fence = "`" * 3  # built in code so this stays one clean string
    reply = f'Sure!\n{fence}json\n{{"question": "q", "answer": "a"}}\n{fence}'
    assert qa.parse_json(reply) == {"question": "q", "answer": "a"}
    assert qa.parse_json("no json here") is None
    assert qa.parse_json("{broken json") is None


def test_splits_are_disjoint_and_cover_all_chapters():
    chapters = [f"Ch{i}" for i in range(19)]
    sp = qa.assign_splits(chapters)
    assert sp["train"] | sp["dev"] | sp["test"] == set(chapters)
    assert not (sp["train"] & sp["test"]) and not (sp["train"] & sp["dev"]) and not (sp["dev"] & sp["test"])
    assert len(sp["test"]) == 5 and len(sp["dev"]) == 2


def test_gen_single_records_gold_chunk():
    items = qa.gen_single(FakeLLM('{"question": "Why?", "answer": "Because."}'), CH, 3)
    assert len(items) == 3
    assert all(it["answerable"] and len(it["gold"]) == 1 for it in items)


def test_gen_single_skips_replies_without_a_question():
    assert qa.gen_single(FakeLLM("sorry, no"), CH, 3) == []


def test_gen_multi_uses_two_different_chapters():
    items = qa.gen_multi(FakeLLM('{"question": "Both?", "answer": "Yes."}'), CH, 2)
    assert items and all(len(it["gold"]) == 2 for it in items)
    by_id = {c["id"]: c for c in CH}
    assert all(by_id[it["gold"][0]]["chapter"] != by_id[it["gold"][1]]["chapter"] for it in items)


def test_unanswerable_skips_topics_the_book_mentions():
    chunks = [{"id": 0, "text": "This book mentions LoRA once.", "chapter": "c", "section": ""}]
    items = qa.gen_unanswerable(FakeLLM('{"questions": ["What is X?"]}'), chunks, ["LoRA", "RLHF"], per_topic=1)
    assert len(items) == 1 and not items[0]["answerable"] and items[0]["a"] is None


def test_with_ids_save_load_roundtrip(tmp_path):
    items = qa.with_ids([{"q": "a"}, {"q": "b"}], "train")
    assert [it["id"] for it in items] == ["train-0", "train-1"]
    qa.save(items, str(tmp_path / "x.jsonl"))
    assert qa.load(str(tmp_path / "x.jsonl")) == items


def test_chunks_in_filters_by_chapter():
    assert [c["id"] for c in qa.chunks_in(CH, {"Ch0"})] == [0, 4]


def test_unanswerable_does_not_skip_a_topic_that_only_appears_inside_a_word():
    chunks = [{"id": 0, "text": "We hold out an exploration set.", "chapter": "c", "section": ""}]
    items = qa.gen_unanswerable(FakeLLM('{"questions": ["What is LoRA?"]}'), chunks, ["LoRA"], per_topic=1)
    assert len(items) == 1  # "lora" is inside "exploration", not a mention of LoRA


class Seq:
    """Returns the queued replies in order, then repeats the last one."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, model, prompt, **kw):
        self.calls += 1
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_bad_question_flags_passage_references_example_numbers_and_code_variables():
    for q in ["What does the passage say about overfitting?",
              "In the context of the passages, why does stratification matter?",
              "Which method is mentioned in Passage 2?",
              "What is shown in Figure 4-2?",
              "What is the value of the third element in the scores array?",
              "What is the accuracy of the model in Example 1-1?",
              "What is the optimal action in state s1?",
              "What does the weights_ array contain after fitting?"]:
        assert qa.bad_question(q), q


def test_bad_question_keeps_ordinary_conceptual_questions():
    for q in ["What is overfitting?",
              "Why does dropout reduce overfitting?",
              "What does the score_samples() method estimate for each instance?",
              "What is the purpose of clustering pixels by color in image segmentation?",
              "What probability distribution does a softmax layer output?"]:
        assert not qa.bad_question(q), q


def test_self_contained_reads_the_judges_yes_or_no():
    assert qa.self_contained(FakeLLM("YES."), "What is overfitting?")
    assert not qa.self_contained(FakeLLM("No"), "What is overfitting?")
    assert not qa.self_contained(FakeLLM(""), "What is overfitting?")  # an empty reply is not a yes


def test_gen_single_oversamples_chunks_until_it_has_n_good_questions():
    bad = '{"question": "What does the passage say?", "answer": "x"}'
    good = '{"question": "What is overfitting?", "answer": "Fitting noise."}'
    llm = Seq(bad, bad, good)  # two bad replies, then good ones forever
    items = qa.gen_single(llm, CH, 2)
    assert len(items) == 2 and llm.calls == 4  # the 2 bad ones were dropped and replaced from other chunks


def test_gen_single_with_the_llm_check_drops_questions_the_judge_rejects():
    reply = '{"question": "What is overfitting?", "answer": "Fitting noise."}'

    class Judge(FakeLLM):
        def chat(self, model, prompt, **kw):
            return "NO" if "Reply YES or NO" in prompt else reply

    assert qa.gen_single(Judge(reply), CH, 3, llm_check=True) == []
    assert len(qa.gen_single(FakeLLM(reply), CH, 3)) == 3  # without the check they are all kept


def test_load_handwritten_csv_marks_blank_answers_unanswerable(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text('question,answer\nWhy bagging?,"Averages models, cutting variance."\nHow does LoRA work?,\n', encoding="utf-8-sig")
    items = qa.load_handwritten_csv(str(p))
    assert [it["id"] for it in items] == ["hw-0", "hw-1"]
    assert items[0]["a"] == "Averages models, cutting variance." and items[0]["answerable"]
    assert items[1]["a"] is None and not items[1]["answerable"] and items[1]["src"] == "hand"


PROSE = "Overfitting happens when a model learns the noise in the training set instead of the underlying pattern. " * 3
CODE = "def play_one_step(env, state, epsilon):\n    q_values = model.predict(state[np.newaxis])\n    next_state, reward, done = env.step(action)\n" * 3


def test_code_heavy_separates_code_from_prose():
    assert qa.code_heavy(CODE)
    assert not qa.code_heavy(PROSE)
    # a prose chunk (~900 chars, like the real ones) that mentions one API name is still prose
    mentions_api = ("The score_samples() method returns the log of the density at each instance. "
                    + "Instances in low density regions are anomalies, and a threshold can come from a percentile. " * 8)
    assert not qa.code_heavy(mentions_api)


def test_bad_question_flags_code_and_example_references_found_in_the_second_hand_check():
    for q in ["What is the purpose of the helper defined in the code snippet?",
              "What is the purpose of the layer defined in the network setup shown?",
              "What is the purpose of picking images in the described clustering approach?",
              "Why is this a risk, according to the provided information?",
              "Why does the training_step() function keep only the maximum?"]:
        assert qa.bad_question(q), q


def test_generators_never_ask_about_code_heavy_chunks():
    prompts = []

    class Spy(FakeLLM):
        def chat(self, model, prompt, **kw):
            prompts.append(prompt)
            return self.reply

    chunks = [{"id": 0, "text": CODE, "chapter": "A", "section": ""}, {"id": 1, "text": PROSE, "chapter": "B", "section": ""},
              {"id": 2, "text": PROSE + " more", "chapter": "C", "section": ""}]
    reply = '{"question": "What is overfitting?", "answer": "Fitting noise."}'
    qa.gen_single(Spy(reply), chunks, 3)
    qa.gen_multi(Spy(reply), chunks, 2)
    assert prompts and not any("play_one_step" in p for p in prompts)

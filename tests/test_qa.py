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

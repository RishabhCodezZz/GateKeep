import pytest

pymupdf = pytest.importorskip("pymupdf")
from gatekeep.corpus import chapter_of, chapter_starts


def make_pdf(path):
    doc = pymupdf.open()
    for _ in range(8):
        doc.new_page()
    doc.set_toc([[1, "Part I. Basics", 2],
                 [2, "Chapter 1. First", 2],
                 [2, "Chapter 2. Second", 4],
                 [1, "Appendix A. Answers", 7],
                 [2, "Chapter 1: Answers for one", 7]])  # colon form belongs to the appendix: must be ignored
    doc.save(str(path))


def test_chapter_starts_reads_numbered_chapters_and_the_back_matter_marker(tmp_path):
    make_pdf(tmp_path / "b.pdf")
    assert chapter_starts(str(tmp_path / "b.pdf")) == [(2, "Chapter 1. First"), (4, "Chapter 2. Second"), (7, "back")]


def test_chapter_of_assigns_pages_to_the_last_chapter_started():
    starts = [(2, "Chapter 1. First"), (4, "Chapter 2. Second"), (7, "back")]
    assert [chapter_of(p, starts) for p in (1, 2, 3, 4, 6, 7, 8)] == [
        "front", "Chapter 1. First", "Chapter 1. First", "Chapter 2. Second", "Chapter 2. Second", "back", "back"]


def test_merge_chunks_gives_one_position_based_id_space_and_keeps_each_source():
    from gatekeep.corpus import merge_chunks
    a, b = [{"id": 0, "text": "a0"}, {"id": 1, "text": "a1"}], [{"id": 0, "text": "b0"}]
    merged = merge_chunks([("Book", a), ("Docs", b)])
    assert [c["id"] for c in merged] == [0, 1, 2] and [c["source"] for c in merged] == ["Book", "Book", "Docs"]
    assert a[0] == {"id": 0, "text": "a0"}  # the inputs are not changed

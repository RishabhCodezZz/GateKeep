import pytest

pytest.importorskip("docling")
pytest.importorskip("transformers")
from gatekeep.corpus import chunk, parse


def test_chunks_carry_headings(tmp_path):
    p = tmp_path / "b.md"
    p.write_text("# Chapter 1\n\n## Intro\n\n" + "word " * 60
                 + "\n\n# Chapter 2\n\n## Other\n\n" + "term " * 60)
    chunks = chunk(parse(str(p)))
    assert len(chunks) >= 2
    assert all(c["chapter"] for c in chunks)
    assert len({c["section"] for c in chunks}) >= 2

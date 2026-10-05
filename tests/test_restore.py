import zipfile

from gatekeep.restore import restore


def make_extracted(root, name, cache_bytes):
    """How Kaggle shows an uploaded zip: it unpacks it into a folder."""
    for sub, fname, size in (("cache", "llm.sqlite", cache_bytes), ("data", "chunks.json", 5), ("results", "rows_V0.jsonl", 3), ("models", "gate.bin", 7)):
        d = root / name / sub
        d.mkdir(parents=True)
        (d / fname).write_bytes(b"x" * size)


def test_restores_from_the_folders_kaggle_makes_when_it_unpacks_the_zip(tmp_path):
    make_extracted(tmp_path / "input", "gatekeep_backup", 10)
    dest = tmp_path / "work"
    dest.mkdir()
    where = restore(str(tmp_path / "input"), str(dest))
    assert where and (dest / "cache" / "llm.sqlite").stat().st_size == 10
    assert (dest / "data" / "chunks.json").exists() and (dest / "results" / "rows_V0.jsonl").exists()
    assert (dest / "models" / "gate.bin").exists()  # fine-tuned gates come back too


def test_picks_the_fullest_backup_when_several_are_attached(tmp_path):
    make_extracted(tmp_path / "input", "gatekeep_backup", 10)       # the old one
    make_extracted(tmp_path / "input", "gatekeep_backup_new", 500)  # more cached replies
    dest = tmp_path / "work"
    dest.mkdir()
    restore(str(tmp_path / "input"), str(dest))
    assert (dest / "cache" / "llm.sqlite").stat().st_size == 500


def test_restores_from_a_real_zip_file_too(tmp_path):
    (tmp_path / "input").mkdir()
    with zipfile.ZipFile(tmp_path / "input" / "gatekeep_backup.zip", "w") as z:
        z.writestr("data/chunks.json", "[]")
        z.writestr("cache/llm.sqlite", "abc")
    dest = tmp_path / "work"
    dest.mkdir()
    assert restore(str(tmp_path / "input"), str(dest))
    assert (dest / "cache" / "llm.sqlite").read_bytes() == b"abc"


def test_returns_none_when_there_is_no_backup(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "work").mkdir()
    assert restore(str(tmp_path / "input"), str(tmp_path / "work")) is None

from pathlib import Path

from tools.convert.proposal import DEFAULT_RANKING, REPOSITORY_ROOT, portable_path


def test_default_ranking_is_repository_relative():
    recorded = portable_path(DEFAULT_RANKING)
    assert recorded == "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    assert str(REPOSITORY_ROOT.parent) not in recorded


def test_path_outside_the_repository_keeps_only_its_name(tmp_path: Path):
    outside = tmp_path / "private" / "ranking.i64"
    assert portable_path(outside) == "ranking.i64"


def test_relative_input_resolves_against_the_working_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert portable_path(Path("a/b.i64"), root=tmp_path) == "a/b.i64"

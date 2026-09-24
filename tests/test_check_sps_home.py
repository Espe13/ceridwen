"""B3-002: ceridwen.check reports $SPS_HOME complete only when the files the forward model
opens are there (it passed an empty nebular/ directory)."""
from ceridwen.check import check_environment


def test_empty_nebular_dir_is_not_all_present(tmp_path, monkeypatch, capsys):
    (tmp_path / "nebular").mkdir()
    monkeypatch.setenv("SPS_HOME", str(tmp_path))
    check_environment(verbose=True)
    out = capsys.readouterr().out
    assert "All components present" not in out
    assert "is it a full FSPS clone?" in out


def test_complete_layout_is_all_present(tmp_path, monkeypatch, capsys):
    for d, f in (("nebular", "ZAU_ND_mist.lines"), ("dust/dustem", "DL07_MW3.1_00.dat"),
                 ("data", "emlines_info.dat")):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / f).write_text("x")
    monkeypatch.setenv("SPS_HOME", str(tmp_path))
    check_environment(verbose=True)
    out = capsys.readouterr().out
    assert "[ok  ] $SPS_HOME" in out

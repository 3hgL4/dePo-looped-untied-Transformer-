"""The table scripts reproduce the paper's numbers from the shipped data and fail on a missing input."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tables"), str(ROOT / "experiments/h2_probing/scripts")]


def test_table1_counts():
    ct = importlib.import_module("capability_table")
    c = ct.compute()
    rates = {key: ct.rate(rs) for key, rs in c.items() if key[0].startswith("N16")}
    assert rates == {("N16", "1e-3", "looped"): "5/5", ("N16", "1e-3", "untied"): "0/5",
                     ("N16", "3e-4", "looped"): "5/5", ("N16", "3e-4", "untied"): "4/5",
                     ("N16", "1e-4", "looped"): "1/5", ("N16", "1e-4", "untied"): "1/5",
                     ("N16 untied 16 layers", "3e-4", "untied"): "1/5",
                     ("N16 auxiliary loss", "1e-3", "untied"): "2/5",
                     ("N16 no tau_t", "1e-3", "looped"): "3/4"}
    assert [r["H_star"] for r in c[("N32", "3e-4", "looped")]] == [6, 0, 1]
    assert [r["H_star"] for r in c[("N32", "3e-4", "untied")]] == [1, 1, 1]
    assert ct.mean_first_solve(c[("N16", "3e-4", "looped")]) == (44.8, 5)
    assert ct.mean_first_solve(c[("N16", "3e-4", "untied")]) == (55.0, 4)


def test_retention():
    values = [round(v, 1) for _, v, _, _ in importlib.import_module("retention").rows()]
    assert values == [2.2, 2.6, 1.7]


def test_capability_table_fails_on_missing_input(tmp_path, monkeypatch):
    ct = importlib.import_module("capability_table")
    monkeypatch.setattr(ct, "EXP", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing input"):
        ct.compute()


def test_retention_fails_on_missing_input(tmp_path, monkeypatch):
    fc = importlib.import_module("figcommon")
    monkeypatch.setattr(fc, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing input"):
        importlib.import_module("retention").rows()

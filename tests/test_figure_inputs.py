"""Every figure builder fails, naming the file, when one of its inputs is missing; none skips silently."""
import importlib
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
H1 = ROOT / "experiments/h1_clean_sweep/scripts"
H2 = ROOT / "experiments/h2_probing/scripts"
sys.path[:0] = [str(H1), str(H2)]

PROBE_BUILDERS = ["build_fig1_teaser", "build_fig4_layouts", "build_fig5_dynamics", "build_fig6_use",
                  "build_appendix_probe_figures", "build_figA5_anomalous_seed"]
RUN_BUILDERS = ["capability_grid", "undertraining_figure", "n32_replication_figure"]


@pytest.mark.parametrize("name", PROBE_BUILDERS)
def test_probe_builder_fails_on_empty_tree(name, tmp_path, monkeypatch):
    fc = importlib.import_module("figcommon")
    monkeypatch.setattr(fc, "DATA", tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="missing input"):
        importlib.import_module(name).main(tmp_path / "out")


@pytest.mark.parametrize("name", RUN_BUILDERS)
def test_run_builder_fails_on_empty_tree(name, tmp_path, monkeypatch):
    mod = importlib.import_module(name)
    attr = "EXP_DIR" if hasattr(mod, "EXP_DIR") else "EXP"
    monkeypatch.setattr(mod, attr, tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="missing input"):
        mod.main(tmp_path / "out")


def test_one_missing_probe_run_fails(tmp_path, monkeypatch):
    """Fig. 5 draws a subset of its runs; removing one that it does not draw must still fail."""
    src = ROOT / "experiments/h2_probing/results"
    dst = tmp_path / "experiments/h2_probing/results"
    for sub in ("geometry_lr1e3_20260920", "fig5_h8_lr1e3_20261009"):
        shutil.copytree(src / sub, dst / sub)
    missing = dst / "fig5_h8_lr1e3_20261009/ouro-4x8_s45_H8/probe_accuracies.npz"  # below 0.95, not drawn
    missing.unlink()
    fc = importlib.import_module("figcommon")
    monkeypatch.setattr(fc, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="ouro-4x8_s45_H8/probe_accuracies.npz"):
        importlib.import_module("build_fig5_dynamics").main(tmp_path / "out")

"""Check that the recipes reproduce the recorded settings of every shipped run.

    uv run python tools/check_recipes.py            # exits non-zero on any mismatch

Every recipe is run with DRY_RUN=1, and each printed command is parsed by the argument parser of
the script it calls. The parsed settings are compared with the shipped records:

  training      experiments/h1_clean_sweep/runs*/<run>/args.json, matched by run directory name
  probes        the "args" block of experiments/h2_probing/results/<out_dir>/probe_results.json
  lens          full_node_lens_summary.json        patching   intervention_summary.json
  directional   directional_probe_summary.json     full answer  <run>_H<H>.json

and every shipped record must be produced by some recipe. Paths (save_dir, checkpoint, out_dir,
device) are compared by run name only. Settings the code no longer has, and settings older
records predate, are compared against the behaviour of the code that wrote them (HISTORICAL
below). Known differences between a record and an uninterrupted rerun are listed, not counted.

--root points the record lookup at another tree with the same layout; --allow-unmatched skips
the check that every record found there is produced by a recipe.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_RECIPES = ["train_grid.sh", "train_rate_controls.sh", "train_aux.sh", "train_notau.sh", "train_n32.sh"]
ANALYSIS_RECIPES = ["probe_prompt_disjoint.sh", "probe_rowsplit.sh", "eval_full_answer.sh", "lens.sh",
                    "patching.sh", "directional.sh"]

IGNORE_TRAIN = {"save_dir", "device", "resume"}
# Training settings that older records lack, or that the trainer no longer has, with the value
# that reproduces what the code of the time did. zero_init is per family: before the flag
# existed the trainer zero-initialized looped models only.
HISTORICAL_TRAIN = {
    "no_timestep_embedding": False, "aux_deep_supervision": False, "aux_layers": "8,16,24",
    "aux_weight": 0.3, "task": "pointer", "cot": False, "keep_ckpt_series": False, "bptt_last_j": None,
    "aux_separate_heads": False, "init_weights": None, "dfs_out_degree": None,
}
IGNORE_PROBE = {"checkpoint", "out_dir", "device"}
HISTORICAL_PROBE = {"task": "pointer", "transfer_matrix": False, "split_by_prompt": False,
                    "complete_answer_eval": False, "per_layer": False, "split_by_correctness": False,
                    "no_timestep_embedding": False}


N_TRAIN = 0  # training commands checked
N_ANALYSIS = 0  # analysis outputs checked


class Parsed(Exception):
    def __init__(self, namespace):
        self.namespace = namespace


def parse(script: str, argv: list[str]) -> dict:
    """Parse argv with the script's own parser and return the namespace, without running it."""
    module = importlib.import_module(script)
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        raise Parsed(original(self, args, namespace))
    argparse.ArgumentParser.parse_args = capture
    saved_argv = sys.argv
    sys.argv = [script] + argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
    except Parsed as parsed:
        return vars(parsed.namespace)
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved_argv
    raise RuntimeError(f"{script} did not parse its arguments")


def dry_run(recipe: str) -> list[list[str]]:
    env = dict(os.environ, DRY_RUN="1", RUNS="RUNS", PROBES="PROBES")
    out = subprocess.run(["bash", str(ROOT / "recipes" / recipe)], env=env, capture_output=True,
                         text=True, check=True).stdout
    return [shlex.split(line[2:]) for line in out.splitlines() if line.startswith("+ ")]


def split_command(tokens: list[str]) -> tuple[str, list[str]]:
    """uv run --no-sync python <script.py | -m module> args... -> (module name, args)."""
    i = tokens.index("python") + 1
    if tokens[i] == "-m":
        return tokens[i + 1], tokens[i + 2:]
    return tokens[i][:-3].replace("/", "."), tokens[i + 1:]


def after(path: str, prefix: str) -> str:
    return path.split(prefix + "/", 1)[1]


def run_of(checkpoint: str) -> str:
    return Path(checkpoint).parent.name


DUPLICATES: list[str] = []  # run names recorded in more than one runs* directory


def find_runs(root: Path) -> dict[str, Path]:
    """Run name -> record directory, over every experiments/h1_clean_sweep/runs* directory.

    When a name repeats, the alphabetically later directory wins (in another tree, a rerun
    directory such as runs_n32_rerun_20261005 sorts after the runs it replaces)."""
    found = {}
    for args in sorted((root / "experiments/h1_clean_sweep").glob("runs*/*/args.json")):
        name = args.parent.name
        if name in found and args.parent not in DUPLICATES:
            DUPLICATES.append(str(args.parent.relative_to(root)))
        found[name] = args.parent
    return found


def check_training(root: Path, mismatches: list, notes: list) -> set[str]:
    runs = find_runs(root)
    produced = set()
    global N_TRAIN
    for recipe in TRAIN_RECIPES:
        for tokens in dry_run(recipe):
            script, argv = split_command(tokens)
            got = parse(script, argv)
            name = Path(got["save_dir"]).name
            produced.add(name)
            N_TRAIN += 1
            if name not in runs:
                mismatches.append(f"{recipe}: {name}: no shipped args.json")
                continue
            rec = json.loads((runs[name] / "args.json").read_text())
            family_zero = got["config_name"].startswith("ouro-")
            for key in sorted(set(got) | set(rec)):
                if key in IGNORE_TRAIN:
                    continue
                if key == "zero_init" and key not in rec:
                    want = family_zero
                elif key == "dfs_out_degree":
                    continue  # read only by the removed --task dfs
                elif key in rec:
                    want = rec[key]
                else:
                    want = HISTORICAL_TRAIN.get(key, "<absent>")
                have = got.get(key, HISTORICAL_TRAIN.get(key, "<absent>"))
                if have != want:
                    mismatches.append(f"{recipe}: {name}: {key} recipe={have!r} record={want!r}")
            if rec.get("resume"):
                notes.append(f"{name}: resumed after an interruption (args.json resume=true) by a trainer "
                             f"that did not yet replay the data rng; an uninterrupted rerun sees different "
                             f"batches after the resume step")
    return {name for name in runs if name not in produced}


def resolve_probe_runtime(got: dict, runs: dict[str, Path]) -> dict:
    """Fill in the settings probe_h2.main resolves from the run's args.json (or its defaults)."""
    train = {}
    if not got["random_init"]:
        train = json.loads((runs[run_of(got["checkpoint"])] / "args.json").read_text())
    got = dict(got)
    if got["use_nope"] is None:
        got["use_nope"] = bool(train.get("use_nope", True))
    if got["zero_init"] is None:
        got["zero_init"] = bool(train.get("zero_init", True))
    got["no_timestep_embedding"] = bool(train.get("no_timestep_embedding", False))
    if got["max_len"] is None:
        got["max_len"] = int(train.get("max_len", 512))
    return got


def compare(label, pairs, mismatches):
    for key, have, want in pairs:
        if have != want:
            mismatches.append(f"{label}: {key} recipe={have!r} record={want!r}")


def check_analysis(root: Path, mismatches: list) -> set[str]:
    res = root / "experiments/h2_probing/results"
    runs = find_runs(root)
    produced = set()
    for recipe in ANALYSIS_RECIPES:
        for tokens in dry_run(recipe):
            script, argv = split_command(tokens)
            got = parse(script, argv)
            if script == "src.probe_h2":
                rel = after(got["out_dir"], "PROBES")
                path = res / rel / "probe_results.json"
                produced.add(str(path.relative_to(res)))
                if not path.exists():
                    mismatches.append(f"{recipe}: {rel}: no shipped probe_results.json")
                    continue
                rec = json.loads(path.read_text())["args"]
                got = resolve_probe_runtime(got, runs)
                if run_of(got["checkpoint"]) != run_of(rec["checkpoint"]):
                    mismatches.append(f"{recipe}: {rel}: checkpoint {got['checkpoint']} vs {rec['checkpoint']}")
                for key in sorted(set(got) | set(rec)):
                    if key in IGNORE_PROBE or key == "config_name" and got[key] == rec.get(key):
                        continue
                    want = rec[key] if key in rec else HISTORICAL_PROBE.get(key, "<absent>")
                    have = got.get(key, HISTORICAL_PROBE.get(key, "<absent>"))
                    if have != want:
                        mismatches.append(f"{recipe}: {rel}: {key} recipe={have!r} record={want!r}")
            elif script == "src.full_node_lens":
                rel = after(got["out_dir"], "PROBES")
                path = res / rel / "full_node_lens_summary.json"
                produced.add(str(path.relative_to(res)))
                if not path.exists():
                    mismatches.append(f"{recipe}: {rel}: no shipped full_node_lens_summary.json")
                    continue
                rec = json.loads(path.read_text())
                compare(f"{recipe}: {rel}", [(k, got[k], rec[k]) for k in
                                             ("config_name", "random_init", "H_eval", "n_samples", "eval_seed")]
                        + [("checkpoint", run_of(got["checkpoint"]), run_of(rec["checkpoint"]))], mismatches)
            elif script == "src.intervention_h2":
                rel = after(got["out_dir"], "PROBES")
                path = res / rel / "intervention_summary.json"
                produced.add(str(path.relative_to(res)))
                if not path.exists():
                    mismatches.append(f"{recipe}: {rel}: no shipped intervention_summary.json")
                    continue
                rec = json.loads(path.read_text())
                compare(f"{recipe}: {rel}", [(k, got[k], rec[k]) for k in
                                             ("mode", "config_name", "random_init", "patch_answer_digit",
                                              "eval_seed", "H")]
                        + [("n_records", got["n_trials"] * got["M"], rec["n_records"]),
                           ("checkpoint", run_of(got["checkpoint"]), run_of(rec["checkpoint"]))], mismatches)
            elif script == "src.directional_probe_h2":
                rel = after(got["out_dir"], "PROBES")
                path = res / rel / "directional_probe_summary.json"
                produced.add(str(path.relative_to(res)))
                if not path.exists():
                    mismatches.append(f"{recipe}: {rel}: no shipped directional_probe_summary.json")
                    continue
                rec = json.loads(path.read_text())
                compare(f"{recipe}: {rel}", [(k, got[k], rec[k]) for k in ("config_name", "N", "H_eval")]
                        + [("n_queries", got["n_samples"] * got["M"], rec["n_queries"]),
                           ("checkpoint", run_of(got["checkpoint"]), run_of(rec["checkpoint"]))], mismatches)
            elif script.endswith("eval_complete_answer"):
                rel = after(str(got["out_dir"]), "PROBES")
                for ckpt in got["checkpoint"]:
                    name = run_of(str(ckpt))
                    path = res / rel / f"{name}_H{got['H_eval']}.json"
                    produced.add(str(path.relative_to(res)))
                    if not path.exists():
                        mismatches.append(f"{recipe}: {rel}/{path.name}: not shipped")
                        continue
                    rec = json.loads(path.read_text())
                    compare(f"{recipe}: {rel}/{path.name}",
                            [(k, got[k], rec[k]) for k in ("H_eval", "n_samples", "eval_seed", "batch_size")]
                            + [("run", name, rec["run"])], mismatches)
            else:
                mismatches.append(f"{recipe}: unrecognised command {script}")
    global N_ANALYSIS
    N_ANALYSIS = len(produced)
    shipped = set()
    for pattern in ("**/probe_results.json", "**/full_node_lens_summary.json", "**/intervention_summary.json",
                    "**/directional_probe_summary.json", "complete_answer_*/*_H*.json"):
        shipped |= {str(p.relative_to(res)) for p in res.glob(pattern)}
    return shipped - produced


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=ROOT, help="tree that holds the records (default: this repo)")
    p.add_argument("--allow-unmatched", action="store_true",
                   help="do not require every record under --root to be produced by a recipe")
    args = p.parse_args()
    mismatches, notes = [], []
    unmatched = check_training(args.root, mismatches, notes)
    unmatched |= check_analysis(args.root, mismatches)
    if not args.allow_unmatched:
        mismatches += [f"run name recorded twice: {d}" for d in sorted(set(DUPLICATES))]
    for line in mismatches:
        print("MISMATCH", line)
    if not args.allow_unmatched:
        for name in sorted(unmatched):
            print("UNMATCHED record without a recipe:", name)
    for line in notes:
        print("NOTE", line)
    n_bad = len(mismatches) + (0 if args.allow_unmatched else len(unmatched))
    print(f"checked {N_TRAIN} training and {N_ANALYSIS} analysis outputs: {len(mismatches)} mismatch(es), "
          f"{len(unmatched)} unmatched record(s), {len(notes)} note(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())

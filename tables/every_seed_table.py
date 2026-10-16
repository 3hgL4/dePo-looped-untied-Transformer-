"""Table A1: first decodable layer of each hop at H=4 and full-answer first-query accuracy, every seed of
Figure A2 (rows from experiments/h2_probing/scripts/build_appendix_probe_figures.py, conditions()).

    uv run python tables/every_seed_table.py [--out tables/every_seed_table.md]

Writes a Markdown table and, beside it, the LaTeX rows (every_seed_table.tex).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/h2_probing/scripts"))
import build_appendix_probe_figures as appendix  # noqa: E402

MODEL = {("a", "ouro-4x8"): "Looped 4x8, 1e-3", ("b", "deep-32"): "Untied, 1e-3", ("c", "deep-32"): "Untied, 3e-4",
         ("d", "deep-32"): "Untied + aux., 1e-3", ("e", "ouro-4x8"): "Looped 4x8, 3e-4"}
PROBES = {"row split, per iteration": "row split", "prompt-disjoint re-probe": "prompt-disjoint",
          "row split": "row split", "prompt-disjoint": "prompt-disjoint"}


def rows():
    out = []
    for r in appendix.table_rows():
        ons = ["--" if o is None else str(o) for o in r["onset_layers_hops_1_3_answer"]]
        out.append([MODEL[(r["panel"], r["family"])], PROBES[r["protocol"]], str(r["seed"]), *ons,
                    f"{r['full_answer_first_query']:.3f}"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "tables/every_seed_table.md")
    args = p.parse_args()
    body = rows()
    head = ["Model, lr", "Probes", "Seed", "Hop 1", "Hop 2", "Hop 3", "Answer", "Accuracy"]
    md = ["# Table A1: first decodable layer (H=4) and full-answer first-query accuracy, every seed", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    args.out.write_text("\n".join(md) + "\n")
    tex, last = [], None
    for r in body:
        model = r[0] if r[0] != last else ""
        if last is not None and r[0] != last:
            tex.append(r"\addlinespace")
        last = r[0]
        tex.append(" & ".join([model] + r[1:]) + r" \\")
    args.out.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

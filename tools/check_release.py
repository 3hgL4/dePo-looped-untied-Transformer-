"""Scan the repository for strings that identify a machine, a user or an internal document.

    uv run python tools/check_release.py             # the working tree
    uv run python tools/check_release.py --history   # also every commit since the submission tag

Exits non-zero and lists every hit. Binary files are skipped; .venv, runs and probe outputs
are ignored. The authors' names in the licence and package metadata are allowed: the user-name
rule matches the lower-case "magnus" that appears in home directories and storage paths only, and
allows the public repository URL.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENY = re.compile(r"torrnode|/storage3|/homes/|/Users/|/private/tmp|scratchpad|physicsofloopedlms|"
                  r"PAPER_STATUS|EXPERIMENT_QUEUE|CAMERA_READY|HANDOVER|CLAUDE\.md|iclr|pooling",
                  re.IGNORECASE)
# The user name, case-sensitive; the author name "Magnus" and the repository URL are allowed.
USER = re.compile(r"\bmagnus\b(?!-sesodia/insidetheloop)")
SKIP_DIRS = {".git", ".venv", "runs", "probe_outputs", "__pycache__", ".pytest_cache", ".mplconfig"}
SELF = Path(__file__).resolve()
TAG = "neurips-2026-submission"


def hit(line: str) -> bool:
    return bool(DENY.search(line) or USER.search(line))


def scan_tree() -> int:
    hits = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts) or path == SELF:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if hit(line):
                hits += 1
                print(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:160]}")
    return hits


def scan_history() -> int:
    """Added lines and commit messages of every commit after the submission tag."""
    log = subprocess.run(["git", "-C", str(ROOT), "log", "-p", "--format=commit %H%n%B", f"{TAG}..HEAD",
                          "--", ".", ":(exclude)tools/check_release.py"],
                         capture_output=True, text=True, check=True).stdout
    hits, commit = 0, None
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1][:7]
            continue
        if line.startswith(("+++", "---")) or line.startswith("-"):
            continue
        if hit(line):
            hits += 1
            print(f"history {commit}: {line.strip()[:160]}")
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--history", action="store_true", help=f"also scan the commits after the {TAG} tag")
    args = p.parse_args()
    hits = scan_tree()
    if args.history:
        hits += scan_history()
    print(f"{hits} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

"""Anonymous release package for double-blind review: code, data, outputs, results.

  python scripts/make_release.py

Builds dist/faultline_anonymous.zip from an allow-list of files (no git history, no lab
journal or paper notes), rewrites the few lines that name the authors or the repository,
and refuses to finish if any file still contains an identifying string. Upload the zip
as supplementary material or to an anonymous repository service.
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "README.md", "LICENSE", "pyproject.toml", "uv.lock", ".gitattributes",
    "faultline", "scripts", "configs", "tests",
    "docs/DATASET.md", "docs/EVAL.md", "docs/RUNNING.md", "docs/ANNOTATION.md",
    "data/faultline_v0.jsonl", "data/faultline_v0.stats.json",
    "annotation/sample_ids.json", "annotation/results",
    "outputs", "results",
]
SKIP = re.compile(r"(make_release\.py$|__pycache__|\.pyc$|\.lock$|queue\.log$|prefetch\.log$|/candidates/|outcomes\.jsonl$|"
                  r"_limit\d+|\.DS_Store$)")
REWRITE = [
    (re.compile(r"Copyright \(c\) (\d{4}) .*"), r"Copyright (c) \1 Anonymous Authors"),
    (re.compile(r"https://github\.com/[\w.-]+/faultline-bench(\.git)?"), "<anonymous repository>"),
    (re.compile(r'git config --global user\.name "[^"]*" && git config --global user\.email "[^"]*"'),
     'git config --global user.name "<your name>" && git config --global user.email "<your email>"'),
    (re.compile(r'git config user\.name "[^"]*" && git config user\.email "[^"]*"'),
     'git config user.name "<your name>" && git config user.email "<your email>"'),
]
# whole words or distinctive strings that would identify the authors or their machines
FORBIDDEN = re.compile(r"\barman\b|shohel|daffodil|gamex|h_server|spark-a405|mds-mac|"
                       r"/users/[a-z]|/home/[a-z]|@gmail\.com|@diu\.", re.I)
TEXT = {".py", ".md", ".toml", ".json", ".jsonl", ".csv", ".tex", ".html", ".sh", ".txt", ".lock", ""}


def files(root):
    for item in INCLUDE:
        p = root / item
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not SKIP.search(str(f.relative_to(root))):
                    yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()

    stage = ROOT / args.out / "faultline"
    if stage.exists():
        shutil.rmtree(stage)
    problems, n = [], 0
    for f in files(ROOT):
        rel = f.relative_to(ROOT)
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in TEXT or f.name == "LICENSE":
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copy2(f, dst)
                n += 1
                continue
            for pat, rep in REWRITE:
                text = pat.sub(rep, text)
            for k, line in enumerate(text.splitlines(), 1):
                m = FORBIDDEN.search(line)
                if m:
                    problems.append(f"{rel}:{k}: {m.group(0)!r}")
            dst.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(f, dst)
        n += 1
    if problems:
        print("identifying strings found; nothing was zipped:")
        print("\n".join(problems[:40]))
        sys.exit(1)
    zpath = ROOT / args.out / "faultline_anonymous.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage.parent))
    print(f"wrote {zpath} ({n} files, {zpath.stat().st_size / 1e6:.1f} MB); no identifying strings found")


if __name__ == "__main__":
    main()

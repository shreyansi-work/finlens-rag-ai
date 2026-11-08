"""Build a clean folder to push to a Hugging Face Space (Docker SDK).

  python scripts/export_space.py ../finlens-space --github https://github.com/<you>/finlens-rag-ai

Why a separate folder: a Space needs a README.md with a YAML header, and Hugging Face rejects pushes that
contain binary files (PNG screenshots) unless they use Git LFS. This copies only what the Dockerfile needs
(no screenshots, no PDFs, no databases: the image regenerates and indexes the sample corpus at build time)
and writes a short Space README. Existing .git in the target is kept, so you can export, commit and push.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY = ["finlens", "web", "scripts", "eval", "Dockerfile", "requirements.txt"]
SKIP_DIRS = {"__pycache__", "results"}   # eval/results are large and not needed to run
SKIP_FILES = {"feedback_candidates.jsonl", "golden.jsonl"}  # golden.jsonl is regenerated at build time

HEADER = """---
title: FinLens
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Agentic RAG analyst for company filings, with citations
---
"""

BODY = """# FinLens

Agentic RAG research analyst for company filings: hybrid retrieval (BM25 + dense, RRF), metadata
self-querying, text-to-SQL with a read-only guard, a calculator tool, a corrective-RAG grader and explicit
abstention. Answers carry page-level citations; click one to see the highlighted source page.

**This demo runs on a synthetic corpus** (five fictional companies, 18 generated PDFs) in rule-based mode
(no LLM). Try: *Compare Nilgiri and Sahyadri operating margins for FY25 and compute the difference.*

Source code, evaluation results (ablation, held-out set, confidence intervals) and limitations: {github}
"""


def copy_tree(src: Path, dst: Path) -> None:
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if item.is_dir() or SKIP_DIRS & set(rel.parts) or item.name in SKIP_FILES or item.suffix == ".pyc":
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst / rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="target folder (usually your cloned Space repo)")
    ap.add_argument("--github", default="https://github.com/shreyansi-work/finlens-rag-ai")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            copy_tree(src, out / name)
        else:
            shutil.copy2(src, out / name)
    (out / "README.md").write_text(HEADER + "\n" + BODY.format(github=args.github), encoding="utf-8")
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"Exported {len(files)} files to {out}")
    binaries = [f for f in files if Path(f).suffix.lower() in {".png", ".jpg", ".pdf", ".db", ".npz"}]
    if binaries:
        print("WARNING: binary files present (Hugging Face needs LFS for these):", binaries)


if __name__ == "__main__":
    main()

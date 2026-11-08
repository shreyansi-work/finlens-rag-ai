"""Run the golden-set evaluation and the ablation study.

  python scripts/evaluate.py                 # full system, all questions -> eval/results/latest.json
  python scripts/evaluate.py --ablation      # ablation + chunking tables -> eval/results/ablation.md
  python scripts/evaluate.py --smoke         # 30-question smoke set (used in CI)

Each ablation row switches on one more technique, so every delta is measured.
Configs that change chunking or embeddings get their own database under
data/eval_dbs/ so they never touch the main index.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from finlens.config import RAW_DIR, settings  # noqa: E402
from finlens.evaluation import evaluate, load_golden  # noqa: E402
from finlens.ingestion.pipeline import ingest_folder  # noqa: E402

RESULTS = ROOT / "eval" / "results"
EVAL_DBS = ROOT / "data" / "eval_dbs"

OFF = dict(use_hybrid=False, use_filters=False, use_rerank=False, use_mmr=False, use_parent_expansion=False,
           use_compression=False, use_context_ordering=False, use_contextualize=False, use_routing=False,
           use_decomposition=False, use_multi_query=False, use_hyde=False, use_step_back=False,
           use_agent=False, use_crag=False, use_reflection=False)

# (label, settings overrides, chunk strategy, contextual headers)
ABLATION = [
    ("Naive baseline (fixed chunks, dense only, single-shot)", OFF, "fixed", False),
    ("Naive + HyDE", {**OFF, "use_hyde": True}, "fixed", False),
    ("Naive + multi-query", {**OFF, "use_multi_query": True}, "fixed", False),
    ("+ hybrid BM25 + RRF", {**OFF, "use_hybrid": True}, "fixed", False),
    ("+ metadata filters (self-querying)", {**OFF, "use_hybrid": True, "use_filters": True}, "fixed", False),
    ("+ reranker + MMR + parent/compression", {**OFF, "use_hybrid": True, "use_filters": True, "use_rerank": True,
                                               "use_mmr": True, "use_parent_expansion": True, "use_compression": True},
     "fixed", False),
    ("+ recursive chunks + contextual headers", {**OFF, "use_hybrid": True, "use_filters": True, "use_rerank": True,
                                                  "use_mmr": True, "use_parent_expansion": True,
                                                  "use_compression": True}, "recursive", True),
    ("+ agentic (routing, SQL, graph, calculator, CRAG, reflection)", {}, "recursive", True),
    ("full + HyDE", {"use_hyde": True}, "recursive", True),
    ("full + multi-query", {"use_multi_query": True}, "recursive", True),
    ("full + step-back", {"use_step_back": True}, "recursive", True),
]


def db_for(strategy: str, headers: bool) -> str:
    """Build (once) an index for a chunking strategy / header setting."""
    if strategy == settings.chunk_strategy and headers == settings.contextual_headers:
        return settings.db_path
    path = EVAL_DBS / f"{strategy}{'_hdr' if headers else ''}.db"
    if not path.exists():
        print(f"  building eval index {path.name} ...")
        ingest_folder(RAW_DIR, settings.with_(db_path=str(path), chunk_strategy=strategy,
                                              contextual_headers=headers), log=lambda *_: None)
    return str(path)


def smoke_set(questions):
    """30 questions covering every type (used by CI)."""
    picked, by_type = [], {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(q)
    while len(picked) < 30 and any(by_type.values()):
        for t in list(by_type):
            if by_type[t] and len(picked) < 30:
                picked.append(by_type[t].pop(0))
    return picked


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def run_ablation(questions):
    lines = ["| Config | Recall@5 | MRR | Faithfulness | Numeric EM | Abstention acc. | False abstention | p95 latency | Cost/query |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for label, overrides, strategy, headers in ABLATION:
        s = settings.with_(**overrides, db_path=db_for(strategy, headers), chunk_strategy=strategy,
                           contextual_headers=headers)
        r = evaluate(s, questions)["summary"]
        rows.append({"config": label, **r})
        lines.append(f"| {label} | {fmt(r['recall@5'])} | {fmt(r['mrr'])} | {fmt(r['faithfulness'])} | "
                     f"{fmt(r['numeric_em'], True)} | {fmt(r['abstention_accuracy'], True)} | "
                     f"{fmt(r['false_abstention_rate'], True)} | "
                     f"{r['p95_latency_ms']} ms | ${r['cost_usd'] or 0:.4f} |")
        print(lines[-1])

    chunk_lines = ["| Chunking strategy | Recall@5 | Recall@20 | MRR | nDCG@10 |", "|---|---|---|---|---|"]
    for strategy in ["fixed", "recursive", "structure", "semantic"]:
        s = settings.with_(db_path=db_for(strategy, True), chunk_strategy=strategy, contextual_headers=True)
        r = evaluate(s, questions, generation=False)["summary"]
        chunk_lines.append(f"| {strategy} | {fmt(r['recall@5'])} | {fmt(r['recall@20'])} | {fmt(r['mrr'])} | "
                           f"{fmt(r['ndcg@10'])} |")
        print(chunk_lines[-1])

    md = ["# Ablation results", "",
          f"Golden set: {len(questions)} questions on the bundled sample corpus. Judge: "
          f"{'LLM' if settings.llm_enabled else 'heuristic (offline)'}. Generated by `python scripts/evaluate.py --ablation`.",
          "", "## Techniques", "", *lines, "", "## Chunking strategies (full retrieval stack)", "", *chunk_lines, ""]
    (RESULTS / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS / 'ablation.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    questions = load_golden()
    if args.smoke:
        questions = smoke_set(questions)
    if args.ablation:
        run_ablation(questions)
        return
    result = evaluate(settings, questions)
    (RESULTS / ("smoke.json" if args.smoke else "latest.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("\nBy question type:")
    for t, s in result["by_type"].items():
        print(f"  {t:14s} n={s['n']:3d}  numeric_em={fmt(s['numeric_em'], True):>6s}  "
              f"faithfulness={fmt(s['faithfulness'])}  recall@5={fmt(s['recall@5'])}")
    print("\nBy source (synthetic = generated from the same data as the PDFs; holdout = hand-written, untuned):")
    for src, s in result["by_source"].items():
        print(f"  {src:10s} n={s['n']:3d}  numeric_em={fmt(s['numeric_em'], True):>6s}  recall@5={fmt(s['recall@5'])}  "
              f"abstention={fmt(s['abstention_accuracy'], True)}  false_abstain={fmt(s['false_abstention_rate'], True)}")
    print("\n95% bootstrap confidence intervals (n = questions behind each metric):")
    for k, v in result["ci95"].items():
        print(f"  {k:22s} n={v['n']:3d}  ci95={v['ci95']}")
    failures = [q for q in result["questions"] if q.get("numeric_em") is False or not q.get("abstain_correct", True)]
    if failures:
        print(f"\n{len(failures)} failing questions:")
        for q in failures[:15]:
            print(f"  {q['id']} [{q['type']}] {q['answer'][:140]}")


if __name__ == "__main__":
    main()

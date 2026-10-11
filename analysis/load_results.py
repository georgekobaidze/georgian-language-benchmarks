"""Collect every run file into two tables for the analysis and the article.

Reads Kaggle runs (results/kaggle/<task>/<version>/<model>/<run id>/, from
`kaggle b t download`) and local test runs (results/local/<task>/<model>/<time>/,
from scripts/local_run.py), and writes:

  analysis/runs.csv   one row per run: task, model, source, score, calls,
                      input/output tokens, cost in dollars, start/end time
  analysis/items.csv  one row per item: id, score, answer, expectation,
                      tokens, cost

Every model call is billed to the same Kaggle model quota, local tests included.

Usage: .venv/bin/python analysis/load_results.py
"""
import csv
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis"


def parse_answer(request: dict) -> str:
    # Structured answers are stored as JSON text, sometimes encoded twice.
    text = request["contents"][-1]["parts"][0].get("text", "")
    try:
        value = json.loads(text)
        if isinstance(value, str):
            value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


def result_value(run: dict):
    results = run.get("results") or [{}]
    r = results[0]
    if "numericResult" in r:
        return r["numericResult"].get("value")
    if "booleanResult" in r:
        return 1.0 if r["booleanResult"] else 0.0
    return None


def usage(run: dict) -> tuple[int, int, int, float]:
    calls = tin = tout = 0
    cost = 0.0
    for c in run.get("conversations", []):
        m = c.get("metrics", {})
        calls += len(c.get("requests", []))
        tin += m.get("inputTokens", 0)
        tout += m.get("outputTokens", 0)
        cost += (int(m.get("inputTokensCostNanodollars", 0)) + int(m.get("outputTokensCostNanodollars", 0))) / 1e9
    return calls, tin, tout, cost


def run_dirs():
    for d in sorted(glob.glob(str(REPO / "results/kaggle/*/*/*/*/"))):
        task, version, model, run_id = Path(d).parts[-4:]
        yield d, task, model, "kaggle", f"v{version}/{run_id}"
    for d in sorted(glob.glob(str(REPO / "results/local/*/*/*/"))):
        task, model, stamp = Path(d).parts[-3:]
        yield d, task, model, "local", stamp


def main():
    runs, items = [], []
    for d, task, model, source, run in run_dirs():
        totals = [0, 0, 0, 0.0]
        score = start = end = None
        n_items = 0
        for f in sorted(glob.glob(os.path.join(d, "*.run.json"))):
            data = json.load(open(f, encoding="utf-8"))
            if "-item-" not in os.path.basename(f):
                score = result_value(data)
                start, end = data.get("startTime"), data.get("endTime")
                continue
            n_items += 1
            calls, tin, tout, cost = usage(data)
            for i, v in enumerate((calls, tin, tout, cost)):
                totals[i] += v
            expectations = [a.get("expectation", "") for a in data.get("assertions", [])]
            requests = [r for c in data.get("conversations", []) for r in c.get("requests", [])]
            items.append({
                "task": task, "model": model, "source": source, "run": run,
                "item": expectations[0].split(":")[0].split(" ")[0] if expectations else "",
                "score": result_value(data),
                "answer": parse_answer(requests[0]) if requests else "",
                "judge": parse_answer(requests[1]) if len(requests) > 1 else "",
                "expectation": " || ".join(expectations),
                "calls": calls, "input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 6),
            })
        runs.append({
            "task": task, "model": model, "source": source, "run": run, "score": score,
            "items": n_items, "calls": totals[0], "input_tokens": totals[1],
            "output_tokens": totals[2], "cost_usd": round(totals[3], 6), "start": start, "end": end,
        })

    for name, rows in (("runs.csv", runs), ("items.csv", items)):
        with open(OUT / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    spent = {s: sum(r["cost_usd"] for r in runs if r["source"] == s) for s in ("kaggle", "local")}
    print(f"{len(runs)} runs, {len(items)} items -> analysis/runs.csv, analysis/items.csv")
    print(f"spent: Kaggle ${spent['kaggle']:.2f}, local tests ${spent['local']:.2f}, total ${sum(spent.values()):.2f}")


if __name__ == "__main__":
    main()

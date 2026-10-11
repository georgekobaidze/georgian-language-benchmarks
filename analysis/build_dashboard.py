"""Build the results dashboard (docs/index.html) from analysis/runs.csv.

Every number on the page is computed here, so the dashboard and the article
use the same figures. Run analysis/load_results.py first to refresh runs.csv.

Usage: .venv/bin/python analysis/build_dashboard.py
"""
import csv
import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "analysis" / "dashboard_template.html"
OUT = REPO / "docs" / "index.html"

# Only runs of this task version count: no thinking, same code for every model.
VERSION = "v4"

TASKS = [
    ("glb-numbers", "Numbers"),
    ("glb-verbs", "Verbs"),
    ("glb-translation", "Translation"),
    ("glb-cloze", "Cloze"),
    ("glb-correction", "Correction"),
]

# Comparable pairs, one Anthropic and one OpenAI model per level.
LEVELS = [
    ("Light", "fast, cheap", "claude-haiku-4-5-20251001", "gpt-5.4-mini-2026-03-17"),
    ("Decent", "everyday workhorse", "claude-sonnet-4-6-default", "gpt-5.4-2026-03-05"),
    ("Top", "flagship", "claude-opus-4-8-default", "gpt-5.5-2026-04-23"),
    ("Newest", "latest flagship", "claude-opus-5-default", "gpt-6-astra"),
]

NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "claude-sonnet-4-6-default": "Claude Sonnet 4.6",
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "claude-opus-4-8-default": "Claude Opus 4.8",
    "gpt-5.5-2026-04-23": "GPT-5.5",
    "claude-opus-5-default": "Claude Opus 5",
    "gpt-6-astra": "GPT-6 Astra",
}

# Tasks whose prompts are identical for every model and make one call per
# item, so their input token counts compare tokenizers directly.
SAME_PROMPT_TASKS = ["glb-numbers", "glb-verbs", "glb-cloze", "glb-correction"]


def scored_items() -> dict[str, int]:
    """Scored items per task, counted from the data files."""
    counts = {}
    with open(REPO / "data" / "numbers.csv", encoding="utf-8") as fh:
        counts["glb-numbers"] = sum(bool(r["expected_cardinal"]) + bool(r["expected_ordinal"]) for r in csv.DictReader(fh))
    for task, name in (("glb-verbs", "verbs.csv"), ("glb-translation", "translation.csv"), ("glb-correction", "correction.csv")):
        with open(REPO / "data" / name, encoding="utf-8") as fh:
            counts[task] = sum(1 for _ in csv.DictReader(fh))
    text = (REPO / "data" / "cloze.yaml").read_text(encoding="utf-8")
    # One "expected" block per text; each gap line looks like "    12: form".
    counts["glb-cloze"] = sum(1 for line in text.splitlines() if line.startswith("    ") and line.strip().split(":")[0].isdigit())
    return counts


def latest_runs() -> dict[tuple[str, str], dict]:
    """The latest completed run of each (model, task) at VERSION."""
    best = {}
    with open(REPO / "analysis" / "runs.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["source"] != "kaggle" or r["state"] != "COMPLETED" or not r["run"].startswith(VERSION + "/"):
                continue
            key = (r["model"], r["task"])
            if key not in best or (r["end"] or "") > (best[key]["end"] or ""):
                best[key] = r
    return best


def main():
    items = scored_items()
    runs = latest_runs()

    models = {}
    for level, _, claude, gpt in LEVELS:
        for slug, vendor in ((claude, "anthropic"), (gpt, "openai")):
            scores, cost, tin, tout = {}, 0.0, 0, 0
            for task, _ in TASKS:
                r = runs.get((slug, task))
                if r:
                    scores[task] = float(r["score"])
                    cost += float(r["cost_usd"])
                    tin += int(r["input_tokens"])
                    tout += int(r["output_tokens"])
            complete = len(scores) == len(TASKS)
            models[slug] = {
                "name": NAMES[slug], "vendor": vendor, "level": level, "scores": scores,
                "average": sum(scores.values()) / len(TASKS) if complete else None,
                "cost": cost if complete else None, "inputTokens": tin, "outputTokens": tout,
            }

    done = [s for s in models if models[s]["average"] is not None]
    best = max(done, key=lambda s: models[s]["average"])
    cheapest = min(done, key=lambda s: models[s]["cost"])

    # Tokenizer gap: input tokens for the same prompts, Claude vs GPT.
    def mean_input(vendor):
        vals = [sum(int(runs[(s, t)]["input_tokens"]) for t in SAME_PROMPT_TASKS)
                for s in done if models[s]["vendor"] == vendor]
        return sum(vals) / len(vals) if vals else None
    gap = (mean_input("anthropic") or 0) / (mean_input("openai") or 1)

    def task_range(task):
        vals = [models[s]["scores"][task] for s in done]
        return min(vals), max(vals)

    tr_lo, tr_hi = task_range("glb-translation")
    nu_lo, nu_hi = task_range("glb-numbers")
    co_lo, co_hi = task_range("glb-correction")
    pairs = [(lv, c, g) for lv, _, c, g in LEVELS if c in done and g in done]
    gpt_wins = sum(models[g]["average"] > models[c]["average"] for _, c, g in pairs)
    lead = ("OpenAI leads at every level so far" if gpt_wins == len(pairs)
            else "Anthropic leads at every level so far" if gpt_wins == 0
            else f"OpenAI leads at {gpt_wins} of {len(pairs)} levels so far")

    # Cost & tokens tab.
    per_call = []
    for task, name in TASKS:
        if task not in SAME_PROMPT_TASKS:
            continue
        row = {"task": name}
        for vendor in ("anthropic", "openai"):
            vals = [int(runs[(s, task)]["input_tokens"]) / int(runs[(s, task)]["calls"])
                    for s in done if models[s]["vendor"] == vendor]
            row[vendor] = sum(vals) / len(vals) if vals else None
        per_call.append(row)

    binary = ["glb-numbers", "glb-verbs", "glb-cloze"]  # tasks scored 0/1 per item
    cost_per_correct = []
    for s in done:
        correct = sum(models[s]["scores"][t] * items[t] for t in binary)
        spent = sum(float(runs[(s, t)]["cost_usd"]) for t in binary)
        cost_per_correct.append({"model": s, "cents": 100 * spent / correct if correct else None,
                                 "correct": round(correct), "of": sum(items[t] for t in binary)})

    spend = {"Final comparison runs": 0.0, "Kaggle's automatic push runs": 0.0,
             "Earlier runs and probes": 0.0, "Local test runs": 0.0}
    with open(REPO / "analysis" / "runs.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            c = float(r["cost_usd"] or 0)
            if r["source"] == "local":
                spend["Local test runs"] += c
            elif r["model"].startswith("gemini-3.7-flash"):
                spend["Kaggle's automatic push runs"] += c
            elif r["run"].startswith(VERSION + "/") and r["model"] in models:
                spend["Final comparison runs"] += c
            else:
                spend["Earlier runs and probes"] += c

    cost_tab = {
        "perTask": {s: {t: float(runs[(s, t)]["cost_usd"]) for t, _ in TASKS if (s, t) in runs} for s in models},
        "perCall": per_call,
        "costPerCorrect": cost_per_correct,
        "spend": [{"label": k, "cost": v} for k, v in spend.items()],
        "spendTotal": sum(spend.values()),
    }

    data = {
        "generated": date.today().isoformat(),
        "costTab": cost_tab,
        "itemsPerModel": sum(items.values()),
        "tasks": [{"id": t, "name": n, "items": items[t]} for t, n in TASKS],
        "levels": [{"name": lv, "hint": hint, "models": [c, g]} for lv, hint, c, g in LEVELS],
        "order": [s for _, _, c, g in LEVELS for s in (c, g)],
        "models": models,
        "kpis": [
            {"label": "Best average so far", "value": f"{models[best]['average']:.2f}",
             "note": f"{models[best]['name']}, {models[best]['level'].lower()} level", "style": "neon"},
            {"label": "Scored items per model", "value": str(sum(items.values())),
             "note": "numbers, verbs, translation, cloze, correction"},
            {"label": "Cheapest full run", "value": f"${models[cheapest]['cost']:.2f}",
             "note": f"{models[cheapest]['name']}, all 5 tasks"},
            {"label": "Georgian token gap", "value": f"{gap:.1f}×",
             "note": f"Claude is billed {gap:.1f} times the input tokens GPT is, for the same Georgian prompts", "style": "hot"},
        ],
        "findings": [
            {"head": "Understanding Georgian is solved.",
             "body": f"Every model translates Georgian into English at {tr_lo:.2f} to {tr_hi:.2f}."},
            {"head": "Producing Georgian is not.",
             "body": f"On numbers the models score {nu_lo:.2f} to {nu_hi:.2f}, on grammar correction {co_lo:.2f} to {co_hi:.2f}."},
            {"head": lead + ",",
             "body": "and a full GPT run costs less, partly because GPT counts the same Georgian text as fewer tokens."},
        ],
        "footnote": f"SCORES 0 TO 1 · KAGGLE BENCHMARKS · TASK {VERSION.upper()} · NO THINKING · ONE RUN PER MODEL · UPDATED {date.today().isoformat()}",
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    # "</" must not appear inside the JSON block, or it could end the script tag.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {len(done)} of {len(models)} models complete, best {models[best]['name']} {models[best]['average']:.3f}")


if __name__ == "__main__":
    main()

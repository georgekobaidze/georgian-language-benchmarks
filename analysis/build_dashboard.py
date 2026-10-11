"""Build the results dashboard (docs/index.html) from analysis/runs.csv.

Every number on the page is computed here, so the dashboard and the article
use the same figures. Run analysis/load_results.py first to refresh runs.csv.

Usage: .venv/bin/python analysis/build_dashboard.py
"""
import csv
import difflib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "analysis" / "dashboard_template.html"
OUT = REPO / "docs" / "index.html"

# Only runs of this task version count: no thinking, same code for every model.
VERSION = "v4"

# The newest flagship pair ran on version 5 with thinking at "low": GPT-6
# Astra cannot switch thinking off, so both models of that pair think a little.
LOW_THINKING_VERSION = "v5"
LOW_THINKING = {"claude-opus-5-default", "gpt-6-astra"}


def version_of(model: str) -> str:
    return LOW_THINKING_VERSION if model in LOW_THINKING else VERSION

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
    """The latest completed run of each (model, task) at that model's version."""
    best = {}
    with open(REPO / "analysis" / "runs.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["source"] != "kaggle" or r["state"] != "COMPLETED" or not r["run"].startswith(version_of(r["model"]) + "/"):
                continue
            key = (r["model"], r["task"])
            if key not in best or (r["end"] or "") > (best[key]["end"] or ""):
                best[key] = r
    return best


def load_items(runs: dict, done: list) -> dict:
    """Item rows of the chosen runs, keyed by (model, task) -> {item id: row}."""
    wanted = {(s, t): runs[(s, t)]["run"] for s in done for t, _ in TASKS}
    out = {}
    csv.field_size_limit(10_000_000)
    with open(REPO / "analysis" / "items.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["model"], r["task"])
            if r["source"] == "kaggle" and wanted.get(key) == r["run"]:
                out.setdefault(key, {})[r["item"]] = r
    return out


def nfc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).translate(DASHES).split())


DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})


def numbers_error(kind: str, form: str, expected: str, answer: str) -> str:
    """Classify a wrong numbers answer into a readable error type."""
    exp_alts = [nfc(e) for e in expected.split("|")]
    a = nfc(answer)
    e = exp_alts[0]
    if kind == "to_digits" and form == "ordinal":
        if e.startswith("მე-") and a.endswith("-ე"):
            return "N-ე instead of მე-N"
        if e.endswith("-ე") and a.startswith("მე-"):
            return "მე-N instead of N-ე"
        if a.replace(" ", "").replace(",", "") in [x.replace(" ", "") for x in exp_alts]:
            return "Thousand separators"
        return "Wrong ordinal in digits"
    if kind == "to_digits":
        return "Wrong value in digits"
    if a == "ერთი " + e:
        return "Added ერთი (one)"
    if "ილიონ" in a and ("ალიონ" in e or "ალონ" in e):
        return "International name for a huge number"
    if form == "ordinal" and "მე" in e:
        if a.replace("მე", "", 1) == e.replace("მე", "", 1):
            return "მე- in the wrong place"
        if a == e.replace("მე", "", 1) or a.replace(" ", "") == e.replace("მე", "", 1).replace(" ", ""):
            return "Missing მე-"
    if a.replace(" ", "") == e.replace(" ", ""):
        return "Spacing between words"
    return "Other wrong form"


def token_list(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", nfc(text))


def correction_breakdown(inp: str, exp: str, ans: str) -> tuple[int, int, int]:
    """(errors needed, errors fixed, correct tokens broken), compared token by token."""
    ti, te, ta = token_list(inp), token_list(exp), token_list(ans)
    # Input tokens that were already correct (unchanged between input and expected).
    correct = set()
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, ti, te, autojunk=False).get_opcodes():
        if op == "equal":
            correct.update(range(i1, i2))
    broken = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ti, ta, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        span = range(i1, i2) if i2 > i1 else range(max(i1 - 1, 0), min(i1 + 1, len(ti)))
        if all(i in correct for i in span):
            broken += max(i2 - i1, j2 - j1)
    needed = edit_distance(ti, te)
    remaining = edit_distance(ta, te)
    fixed = max(0, min(needed, needed - (remaining - broken)))
    return needed, fixed, broken


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def tasks_tab(runs: dict, done: list, models: dict) -> dict:
    rows = load_items(runs, done)

    # Numbers: accuracy by question type, and error types.
    with open(REPO / "data" / "numbers.csv", encoding="utf-8") as fh:
        num = {r["id"]: r for r in csv.DictReader(fh)}
    huge = {str(10 ** k) for k in range(15, 40, 3)}
    cats = [("Digits → words, cardinal", "to_words", "cardinal"), ("Digits → words, ordinal", "to_words", "ordinal"),
            ("Words → digits, cardinal", "to_digits", "cardinal"), ("Words → digits, ordinal", "to_digits", "ordinal")]
    num_grid = []
    for label, kind, form in cats + [("Huge numbers, 10¹⁵ to 10³⁹", None, None)]:
        row = {"label": label, "values": {}}
        for s in done:
            hits = total = 0
            for qid, it in rows[(s, "glb-numbers")].items():
                rid, f = qid.rsplit("-", 1)
                r = num[rid]
                digits = r["input"] if r["kind"] == "to_words" else r["expected_cardinal"]
                if kind is None:
                    if digits not in huge:
                        continue
                elif (r["kind"], f) != (kind, form):
                    continue
                total += 1
                hits += float(it["score"]) == 1.0
            row["values"][s] = hits / total if total else None
            row["n"] = total
        num_grid.append(row)

    errors = {}
    for s in done:
        for qid, it in rows[(s, "glb-numbers")].items():
            if float(it["score"]) == 1.0:
                continue
            rid, f = qid.rsplit("-", 1)
            r = num[rid]
            kind_ = numbers_error(r["kind"], f, r["expected_" + f], it["answer"])
            e = errors.setdefault(kind_, {"count": 0, "example": None})
            e["count"] += 1
            if e["example"] is None:
                e["example"] = {"question": r["input"], "expected": r["expected_" + f].split("|")[0],
                                "answer": it["answer"],
                                "model": models[s]["name"] + (" (thinking: low)" if s in LOW_THINKING else "")}
    num_errors = sorted(({"type": k, **v} for k, v in errors.items()), key=lambda x: -x["count"])

    # Verbs: accuracy per kind, and swapped participants.
    with open(REPO / "data" / "verbs.csv", encoding="utf-8") as fh:
        verbs = {r["id"]: r for r in csv.DictReader(fh)}
    kinds = [("who_to_whom", "Who does what to whom"), ("form", "Build the verb form"), ("preverb", "Choose the preverb")]
    verb_grid, swaps = [], {}
    for k, label in kinds:
        row = {"label": label, "values": {}, "n": sum(1 for v in verbs.values() if v["kind"] == k)}
        for s in done:
            its = [it for vid, it in rows[(s, "glb-verbs")].items() if verbs[vid]["kind"] == k]
            row["values"][s] = sum(float(it["score"]) for it in its) / len(its) if its else None
        verb_grid.append(row)
    for s in done:
        n = 0
        for vid, it in rows[(s, "glb-verbs")].items():
            v = verbs[vid]
            if v["kind"] != "who_to_whom" or float(it["score"]) == 1.0:
                continue
            opts = {o[0]: o[3:] for o in v["choices"].split("|")}
            got = re.fullmatch(r"\(?([A-Da-d])[).]?", it["answer"].strip())
            if not got:
                continue
            chosen, right = opts.get(got.group(1).upper(), ""), opts[v["expected"]]
            cp, rp = chosen.split(" → "), right.split(" → ")
            # Swapped: same people, but the doer and the receiver trade places.
            if len(cp) >= 2 and len(cp) == len(rp) and cp[0] == rp[1] and cp[1] == rp[0]:
                n += 1
        swaps[s] = n
    wrong_who = {s: sum(1 for vid, it in rows[(s, "glb-verbs")].items()
                        if verbs[vid]["kind"] == "who_to_whom" and float(it["score"]) != 1.0) for s in done}

    # Correction: fixed, missed, broken.
    with open(REPO / "data" / "correction.csv", encoding="utf-8") as fh:
        corr = {r["id"]: r for r in csv.DictReader(fh)}
    corr_rows = []
    for s in done:
        needed = fixed = broken = 0
        for cid, it in rows[(s, "glb-correction")].items():
            n_, f_, b_ = correction_breakdown(corr[cid]["input"], corr[cid]["expected"], it["answer"])
            needed += n_
            fixed += f_
            broken += b_
        corr_rows.append({"model": s, "needed": needed, "fixed": fixed, "missed": needed - fixed, "broken": broken})

    # Cloze: every gap, pass or fail, per model.
    gap_re = re.compile(r"(cloze-\d+) \[(\d+)\]: (.*) \(got: (.*)\)$")
    gaps = {}
    for s in done:
        for it in rows[(s, "glb-cloze")].values():
            for part in it["expectation"].split(" || "):
                m = gap_re.match(part)
                if not m:
                    continue
                text, n, accepted, got = m.groups()
                ok = nfc(got) in {nfc(a) for a in accepted.split(" | ")}
                g = gaps.setdefault((text, int(n)), {"text": text, "gap": int(n), "expected": accepted, "results": {}, "answers": {}})
                g["results"][s] = ok
                g["answers"][s] = got
    gap_list = [gaps[k] for k in sorted(gaps)]
    all_wrong = [g for g in gap_list if g["results"] and not any(g["results"].values())]

    return {"numbersGrid": num_grid, "numbersErrors": num_errors, "verbsGrid": verb_grid,
            "swaps": swaps, "wrongWho": wrong_who, "correction": corr_rows,
            "clozeGaps": gap_list, "clozeAllWrong": all_wrong}


# Hand-picked examples. Answers are read from the runs, so new models appear
# automatically. "find" lists word forms to look for in a long answer
# (correction, translation): the first one found is shown as the answer.
EXAMPLES = [
    {"group": "Typical mistakes", "task": "glb-numbers", "item": "numbers-044-ordinal",
     "title": "Eighteenth, in digits", "question": "თვრამეტი (eighteen) → its ordinal in digits",
     "explain": "Georgian forms the ordinal of a simple number with the prefix მე-: მეთვრამეტე, “eighteenth”. In digits the prefix stays in front: მე-18. The suffix form (21-ე) is only for ordinals whose word does not start with მე-, like ოცდამეერთე."},
    {"group": "Typical mistakes", "task": "glb-numbers", "item": "numbers-059-ordinal",
     "title": "Forty-thousandth, in digits", "question": "ორმოცი ათასი (forty thousand) → its ordinal in digits",
     "explain": "The same rule for a big round number: მეორმოციათასე starts with მე-, so it is written მე-40000. Almost every model reached for the suffix instead."},
    {"group": "Typical mistakes", "task": "glb-numbers", "item": "numbers-008-ordinal",
     "title": "Twelve-thousandth, in words", "question": "12000 → its ordinal in Georgian words",
     "explain": "A round number's ordinal is one word: მე- in front, -ე at the end, around the whole number: მე + თორმეტიათას + ე. Models split it, drop the prefix, or attach it to the wrong part."},
    {"group": "Typical mistakes", "task": "glb-numbers", "item": "numbers-021-cardinal",
     "title": "Ten to the 27th, in words", "question": "1 000 000 000 000 000 000 000 000 000 → Georgian words",
     "explain": "The expected Georgian name is ოკტალიონი. Models borrow the international -illion forms instead, and some even pick the wrong power of ten (a sextillion is 10²¹)."},
    {"group": "Typical mistakes", "task": "glb-verbs", "item": "verbs-027",
     "title": "“I fell in love with you”", "question": "შეყვარება (to fall in love), I → you, aorist",
     "explain": "Falling in love is an inversion verb in Georgian: the one who loves is marked like an indirect object, so the form is შემიყვარდი, literally “you became loved to me”. Almost every model built the pattern of an ordinary transitive verb instead (შეგიყვარე)."},
    {"group": "Typical mistakes", "task": "glb-verbs", "item": "verbs-011",
     "title": "Who wrote it?", "question": "დაგიწერია: who did what to whom?",
     "explain": "In the Georgian perfect the markers flip: the გ- in დაგიწერია marks “you” as the writer, so it means “you have (apparently) written it”. The models scattered across the other readings, including “he wrote it to you”."},
    {"group": "Typical mistakes", "task": "glb-verbs", "item": "verbs-044",
     "title": "On or off?", "question": "მან ტელევიზორი ___რთო და დაიძინა. (“He ___ the TV and fell asleep.”)",
     "explain": "The preverb decides the meaning: ჩართო is “switched on”, გამორთო is “switched off”. Only the context, falling asleep, tells which one fits."},
    {"group": "Typical mistakes", "task": "glb-correction", "item": "correction-005",
     "title": "“We sat”", "question": "A paragraph with the planted error ვიჯექით (“we sat”)", "find": ["ვისხედით", "ვსხედით", "ვიჯექით"], "expected": "ვისხედით",
     "explain": "Georgian uses a different verb root for sitting when the subject is plural: “I sat” is ვიჯექი, but “we sat” is ვისხედით. No model fixed it."},
    {"group": "Conventions", "task": "glb-numbers", "item": "numbers-011-cardinal",
     "title": "One million", "question": "1000000 → Georgian words",
     "explain": "A round million is simply მილიონი, the way 1000 is ათასი, not “one thousand”. Most models add ერთი, “one”, as English does."},
]


def examples_tab(runs: dict, done: list, models: dict, gaps: list) -> list:
    rows = load_items(runs, done)
    data = {}
    for name in ("numbers", "verbs", "correction"):
        with open(REPO / "data" / f"{name}.csv", encoding="utf-8") as fh:
            data[name] = {r["id"]: r for r in csv.DictReader(fh)}
    gap_index = {f"{g['text']}:{g['gap']}": g for g in gaps}
    out = []
    for ex in EXAMPLES:
        answers, expected = [], ex.get("expected")
        if ex["task"] == "glb-numbers":
            rid, form = ex["item"].rsplit("-", 1)
            expected = data["numbers"][rid]["expected_" + form].split("|")[0]
        elif ex["task"] == "glb-verbs":
            v = data["verbs"][ex["item"]]
            opts = {o[0]: o[3:] for o in v["choices"].split("|")} if v["choices"] else {}
            expected = opts.get(v["expected"], v["expected"]) if opts else v["expected"]
        for s in done:
            if ex["task"] == "glb-cloze":
                g = gap_index.get(ex["item"])
                if not g or s not in g["answers"]:
                    continue
                expected = g["expected"]
                answers.append({"model": s, "answer": g["answers"][s], "ok": g["results"][s]})
                continue
            it = rows[(s, ex["task"])].get(ex["item"])
            if not it:
                continue
            ans = it["answer"].strip()
            if ex["task"] == "glb-verbs" and opts:
                letter = re.fullmatch(r"\(?([A-Da-d])[).]?", ans)
                ans = opts.get(letter.group(1).upper(), ans) if letter else ans
                ok = float(it["score"]) == 1.0
            elif "find" in ex:
                found = next((f for f in ex["find"] if f in it["answer"]), None)
                ans = (found or "(rewritten)").strip()
                ok = ans == ex["expected"]
            else:
                ok = float(it["score"]) == 1.0
            answers.append({"model": s, "answer": ans, "ok": ok})
        out.append({k: ex[k] for k in ("group", "title", "question", "explain")} | {"expected": expected, "answers": answers})
    return out


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
                "thinking": "low" if slug in LOW_THINKING else "none",
                "average": sum(scores.values()) / len(TASKS) if complete else None,
                "cost": cost if complete else None, "inputTokens": tin, "outputTokens": tout,
            }

    done = [s for s in models if models[s]["average"] is not None]
    # The headline best is among the models without thinking (the newest pair thinks).
    best = max((s for s in done if s not in LOW_THINKING), key=lambda s: models[s]["average"])
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
            elif r["run"].startswith(version_of(r["model"]) + "/") and r["model"] in models:
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

    def rows_in(path):
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in csv.DictReader(fh))
    method_table = [
        {"task": "Numbers", "full": f"{rows_in(REPO / 'archive' / 'numbers_full.csv')} rows",
         "kept": f"{rows_in(REPO / 'data' / 'numbers.csv')} rows, {items['glb-numbers']} questions",
         "rule": "rows where the probe model got at least one answer wrong"},
        {"task": "Verbs", "full": str(rows_in(REPO / "archive" / "verbs_full.csv")), "kept": str(items["glb-verbs"]),
         "rule": "every item the probe got wrong, plus the right answers it struggled with most"},
        {"task": "Translation", "full": str(rows_in(REPO / "archive" / "translation_full.csv")), "kept": str(items["glb-translation"]),
         "rule": "the paragraphs with the most Georgian-specific traps"},
        {"task": "Cloze", "full": "2 texts", "kept": f"2 texts, {items['glb-cloze']} gaps", "rule": "kept in full"},
        {"task": "Correction", "full": str(rows_in(REPO / "archive" / "correction_full.csv")), "kept": str(items["glb-correction"]),
         "rule": "chosen by hand"},
    ]

    data = {
        "generated": date.today().isoformat(),
        "methodTable": method_table,
        "costTab": cost_tab,
        "tasksTab": (tt := tasks_tab(runs, done, models)),
        "examples": examples_tab(runs, done, models, tt["clozeGaps"]),
        "itemsPerModel": sum(items.values()),
        "tasks": [{"id": t, "name": n, "items": items[t]} for t, n in TASKS],
        "levels": [{"name": lv, "hint": hint, "models": [c, g]} for lv, hint, c, g in LEVELS],
        "order": [s for _, _, c, g in LEVELS for s in (c, g)],
        "models": models,
        "kpis": [
            {"label": "Best average, no thinking", "value": f"{models[best]['average']:.2f}",
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
        "footnote": f"SCORES 0 TO 1 · KAGGLE BENCHMARKS · NO THINKING (NEWEST PAIR: LOW) · ONE RUN PER MODEL · UPDATED {date.today().isoformat()}",
        "thinkingNote": "Exception: the newest flagship pair, Claude Opus 5 and GPT-6 Astra, thinks at the lowest level (\"low\"). "
                        "GPT-6 Astra cannot switch thinking off, so both models of the pair get the same small thinking budget. "
                        "Compare them with each other, and with the other levels only with that in mind.",
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    # "</" must not appear inside the JSON block, or it could end the script tag.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {len(done)} of {len(models)} models complete, best {models[best]['name']} {models[best]['average']:.3f}")


if __name__ == "__main__":
    main()

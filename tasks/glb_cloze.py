# glb-cloze: fill the numbered gaps in a Georgian article with the right
# grammatical form of words from a list (the list also has words that fit
# nowhere). One call per text; every gap is scored 0/1 by exact match.
# Score = share of all gaps filled correctly.
#
# Push: scripts/push_task.sh glb-cloze tasks/glb_cloze.py
# Local test: .venv/bin/python scripts/local_run.py tasks/glb_cloze.py
import dataclasses
import json
import os
import time
import unicodedata
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd

try:
    import yaml
except ImportError as e:  # PyYAML ships with Kaggle's Python image
    raise RuntimeError("PyYAML is needed to read cloze.yaml") from e


def data_file(name: str) -> Path:
    # Kaggle mounts the attached dataset somewhere under /kaggle/input (the
    # exact sub-path varies), so search for the file there. Locally, read the
    # repo's data/ folder next to tasks/.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = sorted(kaggle_input.rglob(name))
        if not found:
            raise FileNotFoundError(f"{name} not found under /kaggle/input")
        return found[0]
    return Path(__file__).resolve().parent.parent / "data" / name


PROMPT = (
    "Below is a Georgian text with {n} numbered gaps, [1] to [{n}]. Fill each gap "
    "with the correct grammatical form (case, number, verb person and tense) of "
    "one word from the list. Each needed word is used exactly once; {extra} "
    "words in the list fit nowhere.\n\n"
    "Words: {words}\n\n"
    "Text:\n{text}\n\n"
    "Put the {n} filled-in words in the `answers` field as a list, in gap order: "
    "the first element for [1], the second for [2], and so on. Each element is "
    "only the word form, in Mkhedruli script."
)


def load_items() -> pd.DataFrame:
    texts = yaml.safe_load((data_file("cloze.yaml")).read_text(encoding="utf-8"))
    rows = []
    for t in texts:
        gaps = sorted(int(k) for k in t["expected"])
        assert gaps == list(range(1, len(gaps) + 1)), t["id"]
        # A gap's expected value is one form, or a list of accepted forms.
        expected = [
            v if isinstance(v, list) else [v]
            for v in (t["expected"][g] for g in gaps)
        ]
        rows.append(
            {
                "id": t["id"],
                "text": t["text"].strip(),
                "words": ", ".join(t["words"]),
                "extra": len(t["words"]) - len(gaps),
                # Passed as JSON text so each row is plain strings and numbers.
                "expected": json.dumps(expected, ensure_ascii=False),
            }
        )
    df = pd.DataFrame(rows)
    # Local testing: GLB_LIMIT=N runs only the first N texts.
    limit = int(os.environ.get("GLB_LIMIT", "0"))
    return df.head(limit) if limit else df


ITEMS = load_items()
TOTAL_GAPS = sum(len(json.loads(e)) for e in ITEMS["expected"])


@dataclasses.dataclass
class Answers:
    answers: list[str]


# Dash-like characters that look like "-": hyphen, non-breaking hyphen, figure
# dash, en dash, em dash, minus sign, small and fullwidth hyphen-minus.
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).translate(DASHES)
    return " ".join(text.split())


# Kaggle reserves each call's worst-case cost (maximum output length x price)
# against the quota before sending it, and refuses the call if that doesn't
# fit. A generous limit, far above any answer seen in testing (the longest
# was 26,600 tokens), keeps the reservation affordable for expensive models.
OUTPUT_LIMIT = {"max_completion_tokens": 32000}
# Every model answers directly, without hidden thinking: the same conditions
# for all models, and affordable within the daily quota.


def ask(llm, prompt: str) -> list[str]:
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=Answers, reasoning="none", extra_api_params=OUTPUT_LIMIT).answers
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


@kbench.task(name="glb-cloze-item", store_task=False)
def cloze_item(llm, id: str, text: str, words: str, extra: int, expected: str) -> int:
    gaps = json.loads(expected)
    got = ask(llm, PROMPT.format(n=len(gaps), extra=extra, words=words, text=text))
    correct = 0
    for i, accepted in enumerate(gaps):
        answer = normalize(got[i]) if i < len(got) else ""
        ok = answer in {normalize(a) for a in accepted}
        correct += ok
        kbench.assertions.assert_true(
            ok, expectation=f"{id} [{i + 1}]: {' | '.join(accepted)} (got: {answer or '-'})"
        )
    # Number of correct gaps in this text; the outer task divides by all gaps.
    return correct


@kbench.task(
    name="glb-cloze",
    description=(
        "Georgian cloze: fill numbered gaps in article-style texts with the right "
        "grammatical form of words from a list that includes distractors. "
        "Scored per gap, exact match."
    ),
)
def cloze(llm) -> float:
    runs = cloze_item.evaluate(
        llm=[llm], evaluation_data=ITEMS, n_jobs=2, on_failure="continue"
    )
    # Calls that failed (e.g. the model quota ran out) must not look like a
    # low score: with only 2 texts, any failed text stops without a score.
    errored = len(runs.errored_runs)
    if errored > 0.05 * len(ITEMS):
        raise RuntimeError(f"{errored} of {len(ITEMS)} texts failed with errors; no score reported.")
    correct = sum(r.result for r in runs.completed_runs if r.result is not None)
    return correct / TOTAL_GAPS


cloze.run(kbench.llm)

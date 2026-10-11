# glb-numbers: Georgian numbers, digits to words and words to digits.
# Each row gives a cardinal question and, if expected_ordinal is filled, an
# ordinal question. Every question is a separate call, scored 0/1 by exact
# match. Score = share of questions answered correctly.
#
# Push: scripts/push_task.sh glb-numbers tasks/glb_numbers.py
# Local test on a sample: .venv/bin/python scripts/local_run.py tasks/glb_numbers.py --limit 10
import dataclasses
import os
import time
import unicodedata
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd


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


DIGIT_FORMAT = (
    "Use plain digits with no thousand separators (1000000). Write a minus "
    "sign for negative numbers, a dot for decimals (2.75), a slash for "
    "fractions (3/4) and one space between the whole part and the fraction "
    "of a mixed number (2 3/4). Use a decimal when the fraction is in "
    "tenths, hundredths, thousandths and so on; otherwise use a fraction."
)

PROMPTS = {
    ("to_words", "cardinal"): (
        "Write the number {text} in Georgian words, in Mkhedruli script. "
        "Put only the words in the `answer` field."
    ),
    ("to_words", "ordinal"): (
        "Write the Georgian ordinal number for {text} (as 'fifth' is for 5 "
        "in English) in words, in Mkhedruli script. "
        "Put only the words in the `answer` field."
    ),
    ("to_digits", "cardinal"): (
        "Write this Georgian number in digits: {text}\n"
        + DIGIT_FORMAT
        + " Put only the number in the `answer` field."
    ),
    ("to_digits", "ordinal"): (
        "This is a Georgian cardinal number: {text}\n"
        "Write the corresponding ordinal number in digits, the way it is "
        "written in Georgian. Use plain digits with no thousand separators "
        "(1000000). Put only the answer in the `answer` field."
    ),
}


def load_questions() -> pd.DataFrame:
    rows = pd.read_csv(data_file("numbers.csv"), dtype=str, keep_default_na=False)
    questions = []
    for row in rows.itertuples():
        for form in ("cardinal", "ordinal"):
            expected = getattr(row, f"expected_{form}")
            if expected:
                questions.append(
                    {
                        "qid": f"{row.id}-{form}",
                        "kind": row.kind,
                        "form": form,
                        "text": row.input,
                        "expected": expected,
                    }
                )
    df = pd.DataFrame(questions)
    # Local testing: GLB_LIMIT=N runs a fixed random sample of N questions.
    limit = int(os.environ.get("GLB_LIMIT", "0"))
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    return df


QUESTIONS = load_questions()


@dataclasses.dataclass
class Answer:
    answer: str


# Dash-like characters that look like "-": hyphen, non-breaking hyphen, figure
# dash, en dash, em dash, minus sign, small and fullwidth hyphen-minus.
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(DASHES)
    return " ".join(text.split())


# Kaggle reserves each call's worst-case cost (maximum output length x price)
# against the quota before sending it, and refuses the call if that doesn't
# fit. A generous limit, far above any answer seen in testing (the longest
# was 26,600 tokens), keeps the reservation affordable for expensive models.
OUTPUT_LIMIT = {"max_completion_tokens": 32000}
# Every model answers directly, without hidden thinking: the same conditions
# for all models, and affordable within the daily quota. The exception is
# the newest flagship pair: GPT-6 Astra cannot switch thinking off (its
# minimum is "low"), so both models of that pair think at "low".
LOW_THINKING = ("claude-opus-5", "gpt-6-astra")


def reasoning_for(llm) -> str:
    name = getattr(llm, "model", "") or ""
    return "low" if any(m in name for m in LOW_THINKING) else "none"


def ask(llm, prompt: str) -> str:
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=Answer, reasoning=reasoning_for(llm), extra_api_params=OUTPUT_LIMIT).answer
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


@kbench.task(name="glb-numbers-item", store_task=False)
def number_item(llm, qid: str, kind: str, form: str, text: str, expected: str) -> bool:
    got = normalize(ask(llm, PROMPTS[(kind, form)].format(text=text)))
    # `|` separates alternative correct answers.
    accepted = [normalize(e) for e in expected.split("|")]
    kbench.assertions.assert_in(got, accepted, expectation=f"{qid}: {text} -> {expected}")
    return got in accepted


@kbench.task(
    name="glb-numbers",
    description=(
        "Georgian numbers: digits to words and words to digits, cardinal and "
        "ordinal, from 0 to 10^39 plus fractions and decimals. Exact match."
    ),
)
def numbers(llm) -> float:
    runs = number_item.evaluate(
        llm=[llm], evaluation_data=QUESTIONS, n_jobs=2, on_failure="continue"
    )
    # Calls that failed (e.g. the model quota ran out) must not look like a
    # low score: above 5% failed items, stop without reporting one.
    errored = len(runs.errored_runs)
    if errored > 0.05 * len(QUESTIONS):
        raise RuntimeError(f"{errored} of {len(QUESTIONS)} items failed with errors; no score reported.")
    # A few errored runs (isolated API failures after retries) count as wrong.
    correct = sum(1 for r in runs.completed_runs if r.result is True)
    return correct / len(QUESTIONS)


numbers.run(kbench.llm)

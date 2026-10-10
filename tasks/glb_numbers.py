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

# On Kaggle the dataset is mounted here; locally we read the repo's data/.
KAGGLE_DATA = Path("/kaggle/input/georgian-language-benchmark-data")
DATA = KAGGLE_DATA if KAGGLE_DATA.exists() else Path(__file__).resolve().parent.parent / "data"

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
    rows = pd.read_csv(DATA / "numbers.csv", dtype=str, keep_default_na=False)
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


def ask(llm, prompt: str) -> str:
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=Answer).answer
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
        llm=[llm], evaluation_data=QUESTIONS, n_jobs=4, on_failure="continue"
    )
    # Errored runs (e.g. API failures after retries) count as wrong.
    correct = sum(1 for r in runs.completed_runs if r.result is True)
    return correct / len(QUESTIONS)


numbers.run(kbench.llm)

# glb-correction: fix grammar, spelling and punctuation errors in Georgian
# paragraphs without touching what is already correct.
#
# Scoring, word by word (punctuation marks count as separate tokens):
#   errors_before = edit distance from the input to the expected text
#   errors_after  = edit distance from the model's answer to the expected text
#   score = max(0, (errors_before - errors_after) / errors_before)
# Unfixed errors and changes to correct words both count in errors_after,
# so over-correcting costs points just like missing an error.
# Score = average over paragraphs.
#
# Push: scripts/push_task.sh glb-correction tasks/glb_correction.py
# Local test on a sample: .venv/bin/python scripts/local_run.py tasks/glb_correction.py --limit 3
import dataclasses
import os
import re
import time
import unicodedata
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd

# On Kaggle the dataset is mounted here; locally we read the repo's data/.
KAGGLE_DATA = Path("/kaggle/input/georgian-language-benchmark-data")
DATA = KAGGLE_DATA if KAGGLE_DATA.exists() else Path(__file__).resolve().parent.parent / "data"

PROMPT = (
    "This Georgian paragraph may contain grammar, spelling and punctuation "
    "errors:\n{text}\n\n"
    "Correct the errors. Change only what is wrong: do not rephrase or change "
    "anything that is already correct. If there are no errors, return the "
    "paragraph unchanged. Put only the corrected paragraph in the `answer` field."
)


def load_items() -> pd.DataFrame:
    df = pd.read_csv(DATA / "correction.csv", dtype=str, keep_default_na=False)
    df = df.rename(columns={"input": "text"})
    # Local testing: GLB_LIMIT=N runs a fixed random sample of N rows.
    limit = int(os.environ.get("GLB_LIMIT", "0"))
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    return df


ITEMS = load_items()


@dataclasses.dataclass
class Answer:
    answer: str


# Dash-like characters that look like "-": hyphen, non-breaking hyphen, figure
# dash, en dash, em dash, minus sign, small and fullwidth hyphen-minus.
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})


def tokens(text: str) -> list[str]:
    # Words, and every punctuation mark as its own token.
    text = unicodedata.normalize("NFC", text).translate(DASHES)
    return re.findall(r"\w+|[^\w\s]", text)


def edit_distance(a: list[str], b: list[str]) -> int:
    # Token-level Levenshtein distance: insert, delete or replace one token.
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def ask(llm, prompt: str) -> str:
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=Answer).answer
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


@kbench.task(name="glb-correction-item", store_task=False)
def correction_item(llm, id: str, text: str, expected: str) -> float:
    answer = ask(llm, PROMPT.format(text=text))
    target = tokens(expected)
    before = edit_distance(tokens(text), target)
    after = edit_distance(tokens(answer), target)
    if before == 0:
        # A paragraph without errors: any change is a mistake.
        score = 1.0 if after == 0 else 0.0
    else:
        score = max(0.0, (before - after) / before)
    kbench.assertions.assert_true(
        after == 0,
        expectation=f"{id}: {before} errors before, {after} left after (score {score:.2f})",
    )
    return score


@kbench.task(
    name="glb-correction",
    description=(
        "Georgian grammar correction: fix planted grammar, spelling and "
        "punctuation errors in short paragraphs. Partial credit for each fix; "
        "changing correct text counts against the model."
    ),
)
def correction(llm) -> float:
    runs = correction_item.evaluate(
        llm=[llm], evaluation_data=ITEMS, n_jobs=4, on_failure="continue"
    )
    # Errored runs (e.g. API failures after retries) count as 0.
    return sum(r.result for r in runs.completed_runs if r.result is not None) / len(ITEMS)


correction.run(kbench.llm)

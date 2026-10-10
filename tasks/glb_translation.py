# glb-translation: Georgian -> English translation of short paragraphs.
# Each paragraph is translated by the model under test, then a judge model
# compares its meaning with a native speaker's reference translation on a
# 5-level scale (0, 0.25, 0.5, 0.75, 1). Score = average over paragraphs.
#
# Push: scripts/push_task.sh glb-translation tasks/glb_translation.py
# Local test on a sample: .venv/bin/python scripts/local_run.py tasks/glb_translation.py --limit 3
import dataclasses
import os
import time
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd

# On Kaggle the dataset is mounted here; locally we read the repo's data/.
KAGGLE_DATA = Path("/kaggle/input/georgian-language-benchmark-data")
DATA = KAGGLE_DATA if KAGGLE_DATA.exists() else Path(__file__).resolve().parent.parent / "data"

PROMPT = (
    "Translate this Georgian text into natural English:\n{text}\n\n"
    "Put only the English translation in the `answer` field."
)

JUDGE_PROMPT = """You are grading a translation from Georgian into English.
Compare the MEANING of the candidate translation with the reference translation.
Wording, style and word order do not matter, only whether the same things are said.

Georgian has no grammatical gender. Where the Georgian text does not reveal a
person's gender, any choice of he, she or they is correct and must not lower the score.

Choose one level:
4 = same meaning, nothing missing, wrong or added
3 = a small slip (a minor detail, tense or nuance), the main meaning intact
2 = partly right: one important part is wrong or missing
1 = only fragments right, the overall meaning is lost
0 = wrong, unrelated, or not English

Georgian original:
{source}

Reference translation:
{reference}

Candidate translation:
{candidate}

Put the level (0-4) in `level` and one short sentence explaining it in `reason`."""


def load_items() -> pd.DataFrame:
    df = pd.read_csv(DATA / "translation.csv", dtype=str, keep_default_na=False)
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


@dataclasses.dataclass
class Verdict:
    level: int
    reason: str


def ask(llm, prompt: str, schema):
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=schema)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


@kbench.task(name="glb-translation-item", store_task=False)
def translation_item(llm, id: str, text: str, expected: str) -> float:
    candidate = ask(llm, PROMPT.format(text=text), Answer).answer.strip()
    verdict = ask(
        kbench.judge_llm,
        JUDGE_PROMPT.format(source=text, reference=expected, candidate=candidate),
        Verdict,
    )
    level = min(max(int(verdict.level), 0), 4)
    kbench.assertions.assert_true(
        level == 4, expectation=f"{id}: level {level}/4. {verdict.reason}"
    )
    return level / 4


@kbench.task(
    name="glb-translation",
    description=(
        "Georgian to English translation of short paragraphs full of Georgian-"
        "specific traps (inversion verbs, subjunctives, idioms, no grammatical "
        "gender). A judge model scores meaning against a native speaker's "
        "reference on a 5-level scale."
    ),
)
def translation(llm) -> float:
    runs = translation_item.evaluate(
        llm=[llm], evaluation_data=ITEMS, n_jobs=4, on_failure="continue"
    )
    # Errored runs (e.g. API failures after retries) count as 0.
    return sum(r.result for r in runs.completed_runs if r.result is not None) / len(ITEMS)


translation.run(kbench.llm)

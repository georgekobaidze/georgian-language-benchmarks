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
from kaggle_benchmarks.kaggle.models import load_model


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
    df = pd.read_csv(data_file("translation.csv"), dtype=str, keep_default_na=False)
    df = df.rename(columns={"input": "text"})
    # Local testing: GLB_LIMIT=N runs a fixed random sample of N rows.
    limit = int(os.environ.get("GLB_LIMIT", "0"))
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    return df


ITEMS = load_items()

# A fixed, cheap judge for every model under test, so scores are comparable
# and the judge's cost is known (Kaggle's default judge model is unknown).
JUDGE = load_model("google/gemini-3-flash-preview")


@dataclasses.dataclass
class Answer:
    answer: str


@dataclasses.dataclass
class Verdict:
    level: int
    reason: str


# Kaggle reserves each call's worst-case cost (maximum output length x price)
# against the quota before sending it, and refuses the call if that doesn't
# fit. A generous limit, far above any answer seen in testing (the longest
# was 26,600 tokens), keeps the reservation affordable for expensive models.
OUTPUT_LIMIT = {"max_completion_tokens": 32000}
# Every model answers directly, without hidden thinking: the same conditions
# for all models, and affordable within the daily quota.


def ask(llm, prompt: str, schema):
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=schema, reasoning="none", extra_api_params=OUTPUT_LIMIT)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


@kbench.task(name="glb-translation-item", store_task=False)
def translation_item(llm, id: str, text: str, expected: str) -> float:
    candidate = ask(llm, PROMPT.format(text=text), Answer).answer.strip()
    verdict = ask(
        JUDGE,
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
        llm=[llm], evaluation_data=ITEMS, n_jobs=2, on_failure="continue"
    )
    # Calls that failed (e.g. the model quota ran out) must not look like a
    # low score: above 5% failed items, stop without reporting one.
    errored = len(runs.errored_runs)
    if errored > 0.05 * len(ITEMS):
        raise RuntimeError(f"{errored} of {len(ITEMS)} items failed with errors; no score reported.")
    # A few errored runs (isolated API failures after retries) count as 0.
    return sum(r.result for r in runs.completed_runs if r.result is not None) / len(ITEMS)


translation.run(kbench.llm)

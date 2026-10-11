# glb-verbs: Georgian verb morphology.
#   who_to_whom  read who does what to whom from a verb form (multiple choice)
#   form         build the verb form for given participants and tense (exact match)
#   preverb      pick the preverb that fits a sentence (multiple choice)
# Every row is one call, scored 0/1. Score = share of rows answered correctly.
#
# Push: scripts/push_task.sh glb-verbs tasks/glb_verbs.py
# Local test on a sample: .venv/bin/python scripts/local_run.py tasks/glb_verbs.py --limit 10
import dataclasses
import os
import re
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


TENSES = {
    "present": "present (აწმყო)",
    "imperfect": "imperfect (უწყვეტელი)",
    "future": "future (მყოფადი)",
    "aorist": "aorist (წყვეტილი)",
    "perfect": "perfect (I თურმეობითი)",
}

ARROWS = (
    "Participants are written as subject → indirect object → direct object, "
    "listing only those the verb has, each as a pronoun in the case this verb "
    "form requires. For example, უყიდა ('he bought it for him') is "
    "მან → მას → ის."
)


def load_items() -> pd.DataFrame:
    df = pd.read_csv(data_file("verbs.csv"), dtype=str, keep_default_na=False)
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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(DASHES)
    return " ".join(text.split())


# Kaggle reserves each call's worst-case cost (maximum output length x price)
# against the quota before sending it, and refuses the call if that doesn't
# fit. A generous limit, far above any answer seen in testing (the longest
# was 26,600 tokens), keeps the reservation affordable for expensive models.
OUTPUT_LIMIT = {"max_completion_tokens": 32000}
# Every model answers directly, without hidden thinking: the same conditions
# for all models, and affordable within the daily quota.


def ask(llm, prompt: str) -> str:
    # Nested evaluations can't retry failed items, so retry API errors here.
    for attempt in range(3):
        try:
            return llm.prompt(prompt, schema=Answer, reasoning="none", extra_api_params=OUTPUT_LIMIT).answer
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


def build_prompt(kind: str, text: str, choices: str) -> str:
    # choices is "A) x|B) y|C) z|D) w"; shown one option per line.
    options = choices.replace("|", "\n")
    if kind == "who_to_whom":
        return (
            f"Georgian verb form: {text}\n"
            "Who does what to whom? " + ARROWS + "\n\n"
            + options
            + "\n\nPut only the letter of the correct option in the `answer` field."
        )
    if kind == "preverb":
        return (
            "Fill in the blank (___) in this Georgian sentence with the correct preverb.\n"
            f"Sentence: {text}\n\n"
            + options
            + "\n\nPut only the letter of the correct option in the `answer` field."
        )
    if kind == "form":
        verb, participants, tense = text.split("; ")
        return (
            f"Write the form of the Georgian verb with the verbal noun {verb} "
            f"for {participants}, in the {TENSES[tense]}.\n"
            + ARROWS
            + "\nPut only the verb form, in Mkhedruli script, in the `answer` field."
        )
    raise ValueError(f"unknown kind: {kind}")


def letter_of(answer: str) -> str:
    # Accept "B", "b", "B)", "(B)" or "B."; anything else is wrong.
    m = re.fullmatch(r"\(?([A-Da-d])[).]?", answer.strip())
    return m.group(1).upper() if m else answer.strip()


@kbench.task(name="glb-verbs-item", store_task=False)
def verb_item(llm, id: str, kind: str, text: str, choices: str, expected: str) -> bool:
    reply = ask(llm, build_prompt(kind, text, choices))
    if choices:
        # Multiple choice: expected is the letter of the correct option.
        want = expected
        got = letter_of(reply)
    else:
        want = normalize(expected)
        got = normalize(reply)
    detail = f"{id}: {text} -> {expected}"
    kbench.assertions.assert_equal(want, got, expectation=detail)
    return got == want


@kbench.task(
    name="glb-verbs",
    description=(
        "Georgian verb morphology: who does what to whom from a verb form, "
        "building verb forms by person and tense, and choosing preverbs."
    ),
)
def verbs(llm) -> float:
    runs = verb_item.evaluate(
        llm=[llm], evaluation_data=ITEMS, n_jobs=2, on_failure="continue"
    )
    # Calls that failed (e.g. the model quota ran out) must not look like a
    # low score: above 5% failed items, stop without reporting one.
    errored = len(runs.errored_runs)
    if errored > 0.05 * len(ITEMS):
        raise RuntimeError(f"{errored} of {len(ITEMS)} items failed with errors; no score reported.")
    # A few errored runs (isolated API failures after retries) count as wrong.
    correct = sum(1 for r in runs.completed_runs if r.result is True)
    return correct / len(ITEMS)


verbs.run(kbench.llm)

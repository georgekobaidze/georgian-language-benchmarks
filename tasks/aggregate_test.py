# Smoke test: many items -> one leaderboard score.
# Push: kaggle b t push glb-smoke -f tasks/aggregate_test.py --wait
import dataclasses
import unicodedata

import kaggle_benchmarks as kbench
import pandas as pd

ITEMS = pd.DataFrame(
    [
        {"number": 1, "expected": "ერთი"},
        {"number": 21, "expected": "ოცდაერთი"},
        {"number": 101, "expected": "ას ერთი"},
    ]
)


@dataclasses.dataclass
class Answer:
    words: str


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


# Per-item task. store_task=False keeps it off the task list; its runs are
# still recorded as subruns of the outer task.
@kbench.task(name="glb-smoke-item", store_task=False)
def number_item(llm, number: int, expected: str) -> bool:
    reply = llm.prompt(
        f"Write the number {number} in Georgian words (Mkhedruli script). "
        "Put only the words in the `words` field.",
        schema=Answer,
    )
    got = normalize(reply.words)
    kbench.assertions.assert_equal(
        expected, got, expectation=f"{number} -> {expected}"
    )
    return got == expected


@kbench.task(name="glb-smoke", description="Smoke test: aggregate accuracy over 3 items.")
def smoke(llm) -> float:
    runs = number_item.evaluate(llm=[llm], evaluation_data=ITEMS, on_failure="continue")
    # Errored runs (e.g. API failures) count as wrong.
    correct = sum(1 for r in runs.completed_runs if r.result is True)
    return correct / len(ITEMS)


smoke.run(kbench.llm)

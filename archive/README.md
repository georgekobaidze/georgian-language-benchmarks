# Archive

Full, specialist-checked versions of datasets that were shrunk or removed for the
Kaggle benchmark, because Kaggle's model quota limits how many questions we can run.
They are kept for testing LLMs directly, outside Kaggle. Nothing here is uploaded
to the Kaggle dataset (only `data/` is).

| File | Rows | Kaggle benchmark |
|---|---|---|
| `numbers_full.csv` | 186 | Trimmed to 120 rows in `data/numbers.csv` (easy rows removed, ids renumbered) |
| `verbs_full.csv` | 115 | Trimmed to 50 rows in `data/verbs.csv` (easy rows removed, ids renumbered) |
| `real_world.csv` | 45 | Removed: chat Georgian typed in Latin letters, rewritten in Georgian script |

Formats match `data/`: `|` separates alternative answers or options, multiple-choice
options are `A) ...|B) ...` with the letter in `expected`. Ids are the original ones,
so they don't match the renumbered ids in `data/`.

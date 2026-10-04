#!/usr/bin/env bash
# Push a task to Kaggle, always attaching the benchmark dataset.
# (Pushing without -d silently detaches a previously attached dataset.)
# The dataset must exist first: run scripts/push_data.sh once.
#
# Usage: scripts/push_task.sh <task-slug> <task-file>
# Example: scripts/push_task.sh glb-numbers tasks/numbers.py
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 2 ]; then
  echo "Usage: scripts/push_task.sh <task-slug> <task-file>" >&2
  exit 1
fi
TASK=$1
FILE=$2
[ -f "$FILE" ] || { echo "File not found: $FILE" >&2; exit 1; }
command -v kaggle >/dev/null || { echo "kaggle not found. Run: source .venv/bin/activate" >&2; exit 1; }

DATASET=$(python3 -c "import json; print(json.load(open('data/dataset-metadata.json'))['id'])")

# List my datasets matching the slug. A failed lookup (login, network) stops
# the script, so "missing" can't be confused with "lookup failed".
if ! MINE=$(kaggle datasets list --mine -s "${DATASET#*/}" -v 2>&1); then
  echo "Could not check datasets on Kaggle (login or network problem?):" >&2
  echo "$MINE" >&2
  exit 1
fi
if ! grep -q "^$DATASET," <<< "$MINE"; then
  echo "Dataset $DATASET doesn't exist yet. Run scripts/push_data.sh first." >&2
  exit 1
fi

echo "Pushing $TASK from $FILE with dataset $DATASET"
kaggle b t push "$TASK" -f "$FILE" -d "$DATASET" --wait

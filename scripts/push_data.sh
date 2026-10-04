#!/usr/bin/env bash
# Upload data/ to Kaggle. Creates the dataset on the first run (private),
# and adds a new version on every run after that.
#
# Usage: scripts/push_data.sh "what changed"
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
  echo 'Usage: scripts/push_data.sh "what changed"' >&2
  exit 1
fi
MESSAGE=$1
command -v kaggle >/dev/null || { echo "kaggle not found. Run: source .venv/bin/activate" >&2; exit 1; }

DATASET=$(python3 -c "import json; print(json.load(open('data/dataset-metadata.json'))['id'])")

# List my datasets matching the slug. A failed lookup (login, network) stops
# the script, so "missing" can't be confused with "lookup failed".
if ! MINE=$(kaggle datasets list --mine -s "${DATASET#*/}" -v 2>&1); then
  echo "Could not check datasets on Kaggle (login or network problem?):" >&2
  echo "$MINE" >&2
  exit 1
fi

# Only top-level files in data/ are uploaded (--dir-mode skip ignores subfolders).
if grep -q "^$DATASET," <<< "$MINE"; then
  echo "Updating $DATASET: $MESSAGE"
  kaggle datasets version -p data -m "$MESSAGE" --dir-mode skip
else
  echo "Creating $DATASET (private)"
  kaggle datasets create -p data --dir-mode skip
fi

#!/usr/bin/env bash
# Run a pushed task on a tier of models.
#   dev   2 cheap models, for iterating on a task
#   core  the 8 models analysed in the write-up
#   all   every model in models.txt (once, for the final run)
#
# Usage: scripts/run.sh <task-slug> <dev|core|all> [extra kaggle flags, e.g. --wait]
# Example: scripts/run.sh glb-numbers dev --wait
set -euo pipefail
cd "$(dirname "$0")/.."

DEV="gemini-3.7-flash gpt-oss-20b"
CORE="claude-opus-5-default gpt-5.5-2026-04-23 gemini-3.1-pro-preview grok-4.6
      deepseek-r1-0528 qwen3-235b-a22b-instruct-2507 gpt-oss-120b gpt-oss-20b"

if [ $# -lt 2 ]; then
  echo "Usage: scripts/run.sh <task-slug> <dev|core|all> [extra kaggle flags]" >&2
  exit 1
fi
TASK=$1
TIER=$2
shift 2
command -v kaggle >/dev/null || { echo "kaggle not found. Run: source .venv/bin/activate" >&2; exit 1; }

case "$TIER" in
  dev)  MODELS=$DEV ;;
  core) MODELS=$CORE ;;
  # models.txt is the output of `kaggle b t models`: 2 header lines, slug in column 1.
  all)  MODELS=$(tail -n +3 models.txt | awk 'NF {print $1}') ;;
  *)    echo "Unknown tier: $TIER (use dev, core or all)" >&2; exit 1 ;;
esac

FLAGS=()
for m in $MODELS; do FLAGS+=(-m "$m"); done

COUNT=$(( ${#FLAGS[@]} / 2 ))
if [ "$COUNT" -eq 0 ]; then
  echo "No models found for tier $TIER. Check models.txt." >&2
  exit 1
fi

# Every model costs money; make the big tier a deliberate choice.
if [ "$TIER" = "all" ]; then
  read -r -p "Run $TASK on ALL $COUNT models? Check your spend first. Type 'yes' to continue: " ANSWER
  [ "$ANSWER" = "yes" ] || { echo "Cancelled."; exit 1; }
fi

echo "Running $TASK on $COUNT models ($TIER)"
kaggle b t run "$TASK" "${FLAGS[@]}" "$@"

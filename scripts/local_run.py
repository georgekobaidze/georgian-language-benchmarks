"""Run a task locally through the Kaggle model proxy, saving run files to results/local/.

The library writes run files into the current folder, so this script starts the
task from results/local/<task>/<model>/<date-time>/. Credentials come from .env
(refresh with `kaggle b auth -y`). The model must be in LLMS_AVAILABLE.

Usage:
  .venv/bin/python scripts/local_run.py tasks/glb_numbers.py --limit 10
  .venv/bin/python scripts/local_run.py tasks/glb_numbers.py --limit 10 --model openai/gpt-oss-120b
"""
import argparse
import datetime
import os
import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("task_file", type=Path)
parser.add_argument("--limit", type=int, default=0, help="number of sampled questions (0 = all)")
parser.add_argument("--model", help="model slug; default is LLM_DEFAULT from .env")
args = parser.parse_args()

task_file = args.task_file.resolve()
if not task_file.is_file():
    parser.error(f"file not found: {args.task_file}")

if args.limit:
    os.environ["GLB_LIMIT"] = str(args.limit)

# Importing the library loads .env (overriding the shell) and the default model.
import kaggle_benchmarks as kbench  # noqa: E402
from kaggle_benchmarks.kaggle.models import load_model  # noqa: E402

model = args.model or os.environ["LLM_DEFAULT"]
if args.model:
    # The task file uses kbench.llm, so swap in the requested model.
    kbench.llm = load_model(args.model)

slug = task_file.stem.replace("_", "-")
stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
out = REPO / "results" / "local" / slug / model.split("/")[-1] / stamp
out.mkdir(parents=True)
os.chdir(out)
print(f"Running {slug} on {model}, saving to {out.relative_to(REPO)}")
runpy.run_path(str(task_file), run_name="__main__")

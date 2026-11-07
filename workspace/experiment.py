# Databricks notebook source
# MAGIC %md
# MAGIC # Workspace experiment
# MAGIC
# MAGIC Run your agent through an experiment on this workspace's compute. An experiment
# MAGIC spends no official budget, makes no external model calls and creates no submission.
# MAGIC Its ledger, checkpoints and MLflow runs are written to your **experiment
# MAGIC schema** only, never to the official ledger. The data it reads stays in this
# MAGIC workspace: do not download data, results or checkpoints. Git is the only way work
# MAGIC leaves the workspace, and `.datathon/` is git-ignored so experiment records are
# MAGIC never committed.
# MAGIC
# MAGIC Open this notebook from your team's Git folder and run the cells in order.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Team settings
# MAGIC
# MAGIC Your organizers give you these values. They are names and paths, not credentials:
# MAGIC the catalog, your experiment schema (its name starts with `experiment`), the Volume
# MAGIC for experiment checkpoints, a SQL warehouse id and the Volume that holds the
# MAGIC `datathon` wheel and its dependencies. To experiment on event data instead of the
# MAGIC bundled sample, upload the train-only package files `product_attribute_experiment.jsonl`
# MAGIC and `dataset.json` to a Volume in
# MAGIC your account and set `DATATHON_EXPERIMENT_DATA` to its path (for example
# MAGIC `/Volumes/<catalog>/<schema>/<volume>/event-data`); leave it unset to use the sample.

# COMMAND ----------

import importlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SETTINGS = {
    "DATATHON_EXPERIMENT_CATALOG": "",
    "DATATHON_EXPERIMENT_SCHEMA": "",
    "DATATHON_EXPERIMENT_VOLUME": "",
    "DATATHON_EXPERIMENT_WAREHOUSE": "",
    "DATATHON_WHEEL_VOLUME": "",
}
for name, value in SETTINGS.items():
    os.environ.setdefault(name, value)
# Optional: the Volume folder with the JSONL package and dataset.json (sample data when unset), for example
# os.environ["DATATHON_EXPERIMENT_DATA"] = "/Volumes/<catalog>/<schema>/<volume>/event-data"
if sys.version_info < (3, 11):
    raise RuntimeError(
        "Python 3.11 or later is required: open the Environment panel on the right of the "
        "notebook, pick Environment version 2 or later, apply and run again."
    )
missing = [name for name in SETTINGS if not os.environ.get(name)]
if missing:
    raise RuntimeError("Fill in the team settings above: " + ", ".join(missing))

# The Git folder root is the parent of this notebook's folder.
ROOT = Path(os.getcwd()).resolve().parent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install the harness
# MAGIC
# MAGIC Installs the release the organizer published to the wheel Volume; nothing is
# MAGIC fetched from public package indexes. `requirements.txt` pins the GitLab CI install
# MAGIC only.

# COMMAND ----------

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", os.environ["DATATHON_WHEEL_VOLUME"], "datathon", "databricks-sql-connector",
    ],
    check=True,
)


importlib.invalidate_caches()
# The runtime has already imported its own older typing_extensions; drop it so the
# installed one is loaded instead.
sys.modules.pop("typing_extensions", None)


def datathon(*arguments: str) -> None:
    """Run a `datathon` command in this notebook's process.

    In-process, the Databricks SDK signs in with the notebook's own identity; a child
    process would have no workspace login.
    """
    from datathon.cli import app

    os.chdir(ROOT)
    if app(list(arguments), standalone_mode=False):
        raise RuntimeError("datathon " + arguments[0] + " failed; see the output above")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validate the agent
# MAGIC
# MAGIC The eight contract checks must pass before an experiment is meaningful.

# COMMAND ----------

datathon("validate", ".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Experiment
# MAGIC
# MAGIC Each run needs a new output directory; the timestamp keeps it unique. The
# MAGIC command refuses any schema whose name does not start with `experiment`.

# COMMAND ----------

run_name = datetime.now(timezone.utc).strftime("experiment-%Y%m%dT%H%M%SZ")
output = Path(".datathon") / run_name
data_folder = os.environ.get("DATATHON_EXPERIMENT_DATA", "")
datathon("experiment", ".", "--output", str(output), *(["--data", data_folder] if data_folder else []))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Read the result
# MAGIC
# MAGIC The experiment's ledger, checkpoints and usage records are in your experiment
# MAGIC schema. This notebook does not render checkpoint scores or metrics.

# COMMAND ----------

print(f"Experiment artifacts saved to {output}")

import json
from pathlib import Path

from invoke import task

from spano import plot_params


def load_params(path):
    with open(path, "r") as f:
        return json.load(f)


@task(iterable=["in_files"])
def plot(_, in_files, normalize=False):
    param_groups = {Path(file).name: load_params(file) for file in in_files}
    plot_params(param_groups, normalize=normalize)

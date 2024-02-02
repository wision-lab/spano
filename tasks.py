import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from invoke import task

from spano import Mapping


@task(iterable=["path"])
def plot(_, path, normalize=False):
    fig, axes = plt.subplots(nrows=1, ncols=len(path), sharey=True, squeeze=False)

    if len(path) > 1 and normalize:
        print(
            "WARNING: Normalizing each DOF independently, no direct comparison should be done between plots."
        )

    for p, ax in zip(path, axes.flatten()):
        ax.set_title(Path(p).name)

        with open(p, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if normalize:
                data /= np.abs(data).max(axis=0)

            for i, v in enumerate(np.array(data).T):
                ax.plot(v, label=i, c=f"C{i}")
        else:
            data = {
                int(k): [
                    Mapping.from_params(i).rescale(1 / int(k)).get_params() for i in v
                ]
                for k, v in data.items()
            }
            data = sorted(data.items(), reverse=True)
            steps = np.cumsum([0] + [len(v) for _, v in data])
            data = np.concatenate([v for _, v in data])

            if normalize:
                data /= np.abs(data).max(axis=0)

            for i, v in enumerate(data.T):
                ax.plot(v, c=f"C{i}", label=i)

            for offset in steps[1:-1]:
                ax.axvline(x=offset, c="k", ls="--", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

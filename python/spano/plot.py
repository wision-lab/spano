from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .spano import Mapping

LKParams = List[List[float]] | Dict[int, List[List[float]]]


def plot_params(
    param_groups: Dict[str, LKParams],
    normalize=False,
):
    _, axes = plt.subplots(nrows=1, ncols=len(param_groups), sharey=True, squeeze=False)

    if len(param_groups) > 1 and normalize:
        print(
            "WARNING: Normalizing each DOF independently, no direct comparison should be done between plots."
        )

    for (name, data), ax in zip(param_groups.items(), axes.flatten()):
        ax.set_title(name)

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

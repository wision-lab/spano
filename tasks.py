import json
from pathlib import Path

import matplotlib.pyplot as plt
from natsort import natsorted
from invoke import task


@task()
def plot(c, path, scale=None):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        plt.plot(data)
    else:
        try:
            plt.plot(data[scale])
        except KeyError:
            print(f"Scale not found, expected one of {natsorted(data.keys())}, got {scale}.")
    plt.show()

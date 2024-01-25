from __future__ import annotations

import json
from pathlib import Path
from typing import Self, List

import matplotlib.pyplot as plt
import numpy as np
from invoke import task


class Mapping:
    def __init__(self, mat: np.ndarray, kind: str) -> None:
        self.mat = mat
        self.kind = kind

    @classmethod
    def from_params(cls, params) -> Self:
        match params:
            case [dx, dy]:
                full_params = [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]
                kind = "Translational"
            case [p1, p2, p3, p4, p5, p6]:
                full_params = [p1 + 1.0, p3, p5, p2, p4 + 1.0, p6, 0.0, 0.0, 1.0]
                kind = "Affine"
            case [p1, p2, p3, p4, p5, p6, p7, p8]:
                full_params = [p1 + 1.0, p3, p5, p2, p4 + 1.0, p6, p7, p8, 1.0]
                kind = "Projective"
            case _:
                raise ValueError
        return cls(np.array(full_params).reshape(3, 3), kind)

    @classmethod
    def scale(cls, x, y) -> Self:
        return cls.from_params([x - 1.0, 0.0, 0.0, y - 1.0, 0.0, 0.0])

    @property
    def is_identity(self) -> bool:
        return np.allclose(self.mat, np.eye(3))

    def transform(self, *, lhs=None, rhs=None) -> Self:
        lhs = getattr(lhs, "mat", np.eye(3))
        rhs = getattr(rhs, "mat", np.eye(3))
        return Mapping(lhs @ self.mat @ rhs, "Projective")

    def rescale(self, scale: float) -> Self:
        return self.transform(
            lhs=Mapping.scale(1.0 / scale, 1.0 / scale),
            rhs=Mapping.scale(scale, scale),
        )

    def get_params(self) -> List[float]:
        p = (self.mat / self.mat[-1, -1]).flatten()

        match self.kind:
            case "Translational":
                return [p[2], p[5]]
            case "Affine":
                return [p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5]]
            case "Projective":
                return [p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5], p[6], p[7]]
            case _:
                raise ValueError


@task()
def plot(_, path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        plt.plot(data)
    else:
        data = {
            int(k): [Mapping.from_params(i).rescale(1 / int(k)).get_params() for i in v]
            for k, v in data.items()
        }
        data = sorted(data.items(), reverse=True)
        steps = np.cumsum([0] + [len(v) for _, v in data])

        for offset, (k, v) in zip(steps, data):
            for i, vi in enumerate(np.array(v).T):
                label = i if k == data[0][0] else ""
                plt.plot(np.arange(len(v)) + offset, vi, c=f"C{i}", label=label)

        for offset in steps[1:]:
            plt.axvline(x=offset, c="k", ls="--", alpha=0.2)
        plt.legend()
    plt.show()

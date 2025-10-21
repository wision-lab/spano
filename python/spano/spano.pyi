from __future__ import annotations

from os import PathLike
from typing_extensions import List, Tuple, Dict, Self
from enum import Enum, auto

import numpy.typing as npt

LKParams = Dict[int, List[List[float]]]

class TransformationType(Enum):
    Unknown = auto()
    Identity = auto()
    Translational = auto()
    Homothety = auto()
    Similarity = auto()
    Affine = auto()
    Projective = auto()

    @classmethod
    def from_str(cls, name: str) -> Self: ...
    def num_params(self: Self) -> int: ...
    def to_str(self: Self) -> str: ...

class Mapping:
    mat: npt.NDArray
    kind: str

    @classmethod
    def from_matrix(cls, mat: npt.NDArray, kind: TransformationType) -> Self: ...

    # Some of these are actually staticmethods that return a class
    # instance, this avoids having to have a separate pyo3 wrapper
    @classmethod
    def from_params(cls, params: List[float]) -> Self: ...
    @classmethod
    def scale(cls, x: float, y: float) -> Self: ...
    @classmethod
    def shift(cls, x: float, y: float) -> Self: ...
    @classmethod
    def identity(cls) -> Self: ...
    @staticmethod
    def maximum_extent(
        maps: List[Mapping], sizes: List[Tuple[int, int]]
    ) -> Tuple[npt.NDArray, Mapping]: ...
    @staticmethod
    def interpolate_scalar(
        ts: List[float], maps: List[Mapping], query: float
    ) -> Mapping: ...
    @staticmethod
    def interpolate_array(
        ts: List[float], maps: List[Mapping], query: List[float]
    ) -> List[Mapping]: ...
    @staticmethod
    def accumulate(mappings: List[Mapping]) -> List[Mapping]: ...
    @staticmethod
    def with_respect_to(mappings: List[Mapping], wrt_map: Mapping) -> List[Mapping]: ...
    @staticmethod
    def with_respect_to_idx(
        mappings: List[Mapping], wrt_idx: float
    ) -> List[Mapping]: ...
    @staticmethod
    def accumulate_wrt_idx(
        mappings: List[Mapping], wrt_idx: float
    ) -> List[Mapping]: ...
    def get_params(self) -> List[float]: ...
    def get_params_full(self) -> List[float]: ...
    def inverse(self) -> Self: ...
    def transform(self, *, lhs: Self | None, rhs: Self | None) -> Self: ...
    def rescale(self, scale: float) -> Self: ...
    def warp_points(self, points: npt.NDArray) -> npt.NDArray: ...
    def corners(self, size: Tuple[int, int]) -> npt.NDArray: ...
    def extent(self, size: Tuple[int, int]) -> Tuple[npt.NDArray, npt.NDArray]: ...
    def warp_array(
        self,
        data: npt.NDArray,
        out_size: Tuple[int, int] | None,
        background: List[float] | None,
    ) -> npt.NDArray: ...

def iclk(
    im1: npt.NDArray,
    im2: npt.NDArray,
    init_mapping: Mapping | None = None,
    im1_weights: npt.NDArray | None = None,
    multi: bool = True,
    max_iters: int | None = 250,
    min_dimension: int = 16,
    max_levels: int = 8,
    stop_early: float | None = 1e-3,
    patience: int | None = 10,
    message: bool = False,
) -> Tuple[Mapping, LKParams]: ...
def pairwise_iclk(
    frames: List[npt.NDArray],
    init_mappings: List[Mapping] | None = None,
    multi: bool = True,
    max_iters: int | None = 250,
    min_dimension: int = 16,
    max_levels: int = 8,
    stop_early: float | None = 1e-3,
    patience: int | None = 10,
    message: bool = False,
) -> Tuple[List[Mapping], List[LKParams]]: ...
def img_pyramid(
    im: npt.NDArray, min_dimension: int = 16, max_levels: int = 8
) -> Tuple[npt.NDArray, ...]: ...
def animate_warp(
    img: npt.NDArray,
    params_history: LKParams,
    img_dir: str | PathLike | None = None,
    scale: float = 1.0,
    fps: int = 25,
    step: int = 100,
    out_path: str | PathLike | None = None,
    message: str | None = None,
) -> None: ...
def merge_arrays(
    mappings: List[Mapping],
    frames: List[npt.NDArray],
    size: Tuple[int, int] | None,
) -> npt.NDArray: ...

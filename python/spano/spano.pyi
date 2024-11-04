from os import PathLike
from typing import List, Tuple, Dict, Optional
from typing_extensions import Self
from enum import Enum, auto

import numpy as np

LKParams = Dict[int, List[List[float]]]

class TransformationType(Enum):
    Unknown = auto()
    Identity = auto()
    Translational = auto()
    Homothety = auto()
    Similarity = auto()
    Affine = auto()
    Projective = auto()

    def num_params(self: Self) -> int: ...
    def to_str(self: Self) -> str: ...
    def from_str(name: str) -> Self: ...

class Mapping:
    mat: np.ndarray
    kind: str

    @classmethod
    def from_matrix(cls, mat: np.ndarray, kind: TransformationType) -> Self: ...

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
        maps: List[Self], sizes: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, Self]: ...
    @staticmethod
    def interpolate_scalar(ts: List[float], maps: List[Self], query: float) -> Self: ...
    @staticmethod
    def interpolate_array(
        ts: List[float], maps: List[Self], query: List[float]
    ) -> List[Self]: ...
    @staticmethod
    def accumulate(mappings: List[Self]) -> List[Self]: ...
    @staticmethod
    def with_respect_to(mappings: List[Self], wrt_map: Self) -> List[Self]: ...
    @staticmethod
    def with_respect_to_idx(mappings: List[Self], wrt_idx: float) -> List[Self]: ...
    @staticmethod
    def accumulate_wrt_idx(mappings: List[Self], wrt_idx: float) -> List[Self]: ...
    def get_params(self) -> List[float]: ...
    def get_params_full(self) -> List[float]: ...
    def inverse(self) -> Self: ...
    def transform(self, *, lhs: Optional[Self], rhs: Optional[Self]) -> Self: ...
    def rescale(self, scale: float) -> Self: ...
    def warp_points(self, points: np.ndarray) -> np.ndarray: ...
    def corners(self, size: Tuple[int, int]) -> np.ndarray: ...
    def extent(self, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]: ...
    def warp_array(
        self,
        data: np.ndarray,
        out_size: Optional[Tuple[int, int]],
        background: Optional[List[float]],
    ) -> np.ndarray: ...

def iclk(
    im1: np.ndarray,
    im2: np.ndarray,
    init_mapping: Optional[Mapping] = None,
    im1_weights: Optional[np.ndarray] = None,
    multi: bool = True,
    max_iters: Optional[int] = 250,
    min_dimension: int = 16,
    max_levels: int = 8,
    stop_early: Optional[float] = 1e-3,
    patience: Optional[int] = 10,
    message: bool = False,
) -> Tuple[Mapping, LKParams]: ...
def pairwise_iclk(
    frames: List[np.ndarray],
    init_mappings: Optional[List[Mapping]] = None,
    multi: bool = True,
    max_iters: Optional[int] = 250,
    min_dimension: int = 16,
    max_levels: int = 8,
    stop_early: Optional[float] = 1e-3,
    patience: Optional[int] = 10,
    message: bool = False,
) -> Tuple[List[Mapping], List[LKParams]]: ...
def img_pyramid(
    im: np.ndarray, min_dimension: int = 16, max_levels: int = 8
) -> Tuple[np.ndarray, ...]: ...
def animate_warp(
    img: np.ndarray,
    params_history: LKParams,
    img_dir: Optional[PathLike] = None,
    scale: float = 1.0,
    fps: int = 25,
    step: int = 100,
    out_path: Optional[PathLike] = None,
    message: Optional[str] = None,
) -> None: ...

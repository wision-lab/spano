from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .spano import Mapping

LKParams = Dict[int, List[List[float]]]


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
        data = {
            int(k): [Mapping.from_params(i).rescale(1 / int(k)).get_params() for i in v]
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
        ax.set_title(name)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


# def animate_warp(
#     img,
#     params_history: LKParams,
#     img_dir: Optional[PathLike] = None,
#     scale: float = 1.0,
#     fps: int = 25,
#     step: int = 100,
#     out_path: Optional[PathLike] = None,
#     message: Optional[str] = None,
# ):
#     if img_dir is None and out_path is None:
#         raise ValueError("At least one of `img_dir`, `out_path` needs to be specified!")

#     if img.dtype != np.uint8:
#         raise ValueError(f"Image array must have np.uint8 dtype, instead got {img.dtype}.")

#     img_dir_context = (
#         tempfile.TemporaryDirectory()
#         if img_dir is None
#         else contextlib.nullcontext(img_dir)
#     )

#     with img_dir_context as img_dir:
#         if isinstance(params_history, list):
#             _animate_warp(
#                 img,
#                 params_history,
#                 img_dir,
#                 scale=scale,
#                 fps=fps,
#                 step=step,
#                 out_path=out_path,
#                 message=message,
#             )
#         else:
#             _animate_hierarchical_warp(
#                 img,
#                 params_history,
#                 img_dir,
#                 scale=scale,
#                 fps=fps,
#                 step=step,
#                 out_path=out_path,
#                 message=message,
#             )

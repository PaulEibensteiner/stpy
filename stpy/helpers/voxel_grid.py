from typing import List, Optional, Union


import torch
from torch import Tensor

from torch_cluster import grid_cluster


def _calculate_voxel_size(x: Tensor, n_voxels: int) -> float:
    data_range = x.max(dim=0).values - x.min(dim=0).values
    total_volume = torch.prod(data_range)
    voxel_volume = total_volume / n_voxels
    voxel_size = voxel_volume ** (1 / x.shape[1])
    return voxel_size.item()


def voxel_grid(
    x: Tensor,
    size: Union[float, Tensor, None] = None,
    approx_n_voxels: int | None = None,
) -> Tensor:
    # approx_n_voxels is only correct if the input domain is a (hyper) cube
    # in every other case the result will either be more or less

    if size is None:
        assert approx_n_voxels is not None, "One of size, n_voxels must be given"
        size = _calculate_voxel_size(x, approx_n_voxels)

    if isinstance(size, float):
        size = torch.full([x.shape[1]], size)
    indices = grid_cluster(x, size).unsqueeze(1).expand(-1, x.shape[1])
    out = torch.full(
        [indices.max() + 1, x.shape[1]], torch.nan, dtype=x.dtype, device=x.device
    )
    averaged = out.scatter_reduce(0, indices, x, reduce="mean", include_self=False)
    return averaged[~torch.isnan(averaged).any(dim=1)]


if __name__ == "__main__":

    # Example usage of voxel_grid
    x = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [2.1, 2.2, 2.3],
            [3, 3, 3],
        ]
    )
    size = 1.0

    result = voxel_grid(x, approx_n_voxels=3)
    print(result)

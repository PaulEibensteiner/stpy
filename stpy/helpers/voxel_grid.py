from typing import List, Optional, Union


import torch
from torch import Tensor

from torch_cluster import grid_cluster


def _get_n_voxels(x, size: float):
    size = torch.full([x.shape[1]], size)
    indices = grid_cluster(x, size)
    return indices.unique().numel()


def voxel_grid(
    x: Tensor,
    size: Union[float, Tensor, None] = None,
    max_n_voxels: int | None = None,
) -> Tensor:

    # Do binary search to find the right voxel size that yields <= max_n_voxels
    if size is None:
        assert max_n_voxels is not None, "One of size, n_voxels must be given"
        max_size = (x.max(dim=0).values - x.min(dim=0).values).max().item()
        tol = max_size / 1e7
        low, high = 0, max_size
        while high - low > tol:
            mid = (low + high) / 2
            n_voxels = _get_n_voxels(x, mid)
            if n_voxels > max_n_voxels:
                low = mid
            else:
                high = mid
        size = high

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

    result = voxel_grid(x, max_n_voxels=3)
    print(result)

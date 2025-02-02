from typing import overload
from torch.multiprocessing import Pool, set_start_method
from os import cpu_count
import line_profiler
import torch
import numpy as np
from scipy.spatial import Delaunay, cKDTree

shared_triangulation: Delaunay | None
xtree: cKDTree


def _initialize(tri: Delaunay, tree: cKDTree):
    global shared_triangulation
    global xtree
    shared_triangulation = tri
    xtree = tree


def _find_exact_or_simplex_batch(batch: np.ndarray):
    distances, idx = xtree.query(batch, k=1, distance_upper_bound=1e-7)
    exact_match_mask = distances <= 1e-7
    batch_remaining = batch[~exact_match_mask]

    simplices = shared_triangulation.find_simplex(batch_remaining)
    outside_conv_hull_mask = simplices < 0

    simplices_remaining = simplices[~outside_conv_hull_mask]
    exact_match_mask[~exact_match_mask] = outside_conv_hull_mask

    idx = idx[exact_match_mask]
    if len(simplices_remaining) > 0:
        no_match_mask = idx == len(xtree.data)
        if no_match_mask.any():
            _, idx_no_match = xtree.query(batch[exact_match_mask][no_match_mask], k=1)
            idx[no_match_mask] = idx_no_match

    return idx, simplices_remaining, exact_match_mask


class InterpolatorArray:

    def __init__(self, x: torch.Tensor, phi: torch.Tensor, m: int, num_cpu_cores=None):
        # Ensure x is on CPU for Delaunay
        x_cpu = x.cpu().numpy()

        # Build the Delaunay triangulation on CPU
        tri = Delaunay(x_cpu)
        xtree = cKDTree(x_cpu)
        if num_cpu_cores is None:
            num_cpu_cores = cpu_count()
        self.num_cpu_cores = num_cpu_cores
        pool = Pool(num_cpu_cores, _initialize, [tri, xtree])
        self.interpolators = [
            InterpolatorND(x, phi[:, j], tri, xtree, pool, num_cpu_cores)
            for j in range(m)
        ]
        self.pools = [pool]

    def __call__(self, j: int, q: torch.Tensor):
        return self.interpolators[j](q).view(-1, 1)

    def add(self, x: torch.Tensor, phi: torch.Tensor, m: int):
        x_cpu = x.cpu().numpy()
        tri = Delaunay(x_cpu)
        xtree = cKDTree(x_cpu)
        pool = Pool(self.num_cpu_cores, _initialize, [tri, xtree])
        self.interpolators.extend(
            [
                InterpolatorND(x, phi[:, j], tri, xtree, pool, self.num_cpu_cores)
                for j in range(m)
            ]
        )
        self.pools.append(pool)

    def __del__(self):
        for pool in self.pools:
            pool.close()
            pool.join()


class InterpolatorND:
    """
    Piecewise linear interpolator for N-dimensional data using Delaunay triangulation.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        tri=None,
        xtree=None,
        pool=None,
        num_cpu_cores=None,
    ):
        """
        Args:
            x: (N, D) tensor of input points in D-dimensional space.
            y: (N,) tensor of function values at those points.
            tri: Precomputed Delaunay triangulation.
            xtree: Precomputed cKDTree for nearest neighbor search.
            pool: Optional multiprocessing pool.
            num_cpu_cores: Number of CPU cores to use for parallel processing.
        """
        assert y.dtype == torch.float64

        if tri is not None:
            # Use the provided Delaunay triangulation and cKDTree
            self.tri = tri
            self.xtree = xtree
            self.pool = pool
            self.own_pool = False
            self.num_cpu_cores = num_cpu_cores
        else:
            # Ensure x is on CPU for Delaunay
            x_cpu = x.detach().cpu().numpy()

            # Build the Delaunay triangulation on CPU
            self.tri = Delaunay(x_cpu)
            xtree = cKDTree(x_cpu)

            if num_cpu_cores is None:
                num_cpu_cores = cpu_count()
            self.num_cpu_cores = num_cpu_cores
            self.pool = Pool(num_cpu_cores, _initialize, [self.tri, xtree])
            self.own_pool = True

        self.x = x
        self.y = y

        # Convert triangle simplices to a torch tensor
        simplices = torch.tensor(self.tri.simplices, device=y.device)
        self.simplices = simplices  # Shape: (M, D+1), M = # of simplices

        # Gather simplex vertex positions and function values
        self.tri_pts = x[simplices]  # Shape: (M, D+1, D)
        self.tri_y = y[simplices]  # Shape: (M, D+1)

        # Precompute matrices for barycentric transformation
        v0 = self.tri_pts[:, 0, :]  # First vertex of each simplex
        T = self.tri_pts[:, 1:, :] - v0[:, None, :]  # (M, D, D)
        T = T.transpose(-1, -2)
        self.T_inv = torch.inverse(T)  # (M, D, D)
        self.v0 = v0  # Store v0 for barycentric computation

    def __del__(self):
        if self.own_pool:
            self.pool.close()
            self.pool.join()

    def __call__(self, xp: torch.Tensor) -> torch.Tensor:
        """
        Interpolate y-values at query points xp. Does not keep order of points the same!

        Args:
            xp: (B, D) tensor of query points in D-dimensional space.

        Returns:
            out: (B,) tensor of interpolated values.
        """

        xp_cpu = xp.cpu().numpy()  # (B, D)

        # 1) Use Delaunay.find_simplex on CPU to find simplices
        # simplex_idx = self.tri.find_simplex(xp_cpu)  # (B,)

        # Split xp_cpu into batches for parallel processing
        batches = np.array_split(xp_cpu, self.num_cpu_cores)
        # Use multiprocessing to parallelize find_simplex
        results = self.pool.map_async(
            _find_exact_or_simplex_batch, [batch for batch in batches]
        ).get(timeout=10)
        # Concatenate the results back into a single array
        # results = [(out_exact_matches0, xp0, simplices0), (out_exact_matches1, xp1, simplices1), ...]
        exact_matches_idx_list = []
        exact_match_mask_list = []
        simplices_list = []
        for exact_matches_idx, simplices, exact_match_mask in results:
            exact_matches_idx_list.append(exact_matches_idx)
            exact_match_mask_list.append(exact_match_mask)
            simplices_list.append(simplices)

        exact_matches_idx = np.concatenate(exact_matches_idx_list)
        exact_matches_idx = torch.tensor(exact_matches_idx)
        exact_matches_y = self.y[exact_matches_idx]
        if len(exact_matches_y) == len(xp):
            return exact_matches_y

        exact_match_mask = np.concatenate(exact_match_mask_list)
        exact_match_mask = torch.tensor(exact_match_mask)

        simplices_remaining = np.concatenate(simplices_list)
        simplices_remaining = torch.tensor(simplices_remaining)  # (B,)

        xp_remaining = xp[~exact_match_mask]

        # p - v0: (Bv, D)
        p_v0 = xp_remaining - self.v0[simplices_remaining]

        # alpha = T_inv @ (p - v0): (Bv, D)
        T_inv_local = self.T_inv[simplices_remaining]  # (Bv, D, D)
        # Batched Matrix multiplication, but T_inv_local is transposed
        bary_coords = torch.einsum("bij, bj -> bi", T_inv_local, p_v0)  # (Bv, D)

        # Compute last barycentric coordinate
        bary_coords = torch.cat(
            [1 - bary_coords.sum(dim=-1, keepdim=True), bary_coords], dim=-1
        )  # (Bv, D+1)

        # 5) Interpolate y-values using barycentric coordinates
        tri_y_local = self.tri_y[simplices_remaining]  # (Bv, D+1)
        out_interpolated = (bary_coords * tri_y_local).sum(dim=-1)  # (Bv,)

        # 6) Store results for valid points
        result = torch.empty(len(xp), dtype=self.y.dtype, device=self.y.device)
        result[exact_match_mask] = exact_matches_y
        result[~exact_match_mask] = out_interpolated
        return result


def plot_simple_function():

    # Define a simple 2D function
    def simple_function(x, y):
        return np.sin(np.pi * x) * np.cos(np.pi * y)

    # Generate a grid of points for the original function
    n_points = 21  # Number of points along each axis
    x_vals = np.linspace(0, 1, n_points)
    y_vals = np.linspace(0, 1, n_points)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = simple_function(x_grid, y_grid)  # Compute function values

    # Flatten the grid for input to the interpolator
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()

    # Convert to PyTorch tensors
    x_train = torch.tensor(
        np.column_stack((x_flat, y_flat)), dtype=torch.float64, device="cuda"
    )

    y_train = torch.tensor(z_flat, dtype=torch.float64, device="cuda")

    # Create the interpolator
    interpolator = InterpolatorND(x_train, y_train, num_cpu_cores=1)

    # Generate slightly offset query points
    n_query = 21
    x_query_vals = np.linspace(0.010, 1.01, n_query)
    y_query_vals = np.linspace(0.010, 1.01, n_query)
    # x_query_vals = np.array([0.31])
    # y_query_vals = np.array([0.01])
    x_query_grid, y_query_grid = np.meshgrid(x_query_vals, y_query_vals)
    x_query_grid = np.concat(
        [np.linspace(0.0, 1.0, n_query).reshape(1, -1), x_query_grid]
    )
    y_query_grid = np.concat([np.zeros([1, n_query]), y_query_grid])
    x_query_flat = x_query_grid.flatten()
    y_query_flat = y_query_grid.flatten()

    # Convert query points to PyTorch tensors
    x_query = torch.tensor(
        np.column_stack((x_query_flat, y_query_flat)),
        dtype=torch.float64,
        device="cuda",
    )

    # Perform interpolation
    z_query = interpolator(x_query).cpu().numpy()  # Interpolated values

    # Plot the original function as a scatter plot
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(x_flat, y_flat, c=z_flat, cmap="viridis", s=40)
    plt.title("Original Function")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(x_query_flat, y_query_flat, c=z_query, cmap="viridis", s=200)
    plt.scatter(
        x_flat,
        y_flat,
        c=z_flat,
        cmap="viridis",
        s=200,
    )
    plt.title("Interpolated Function with Original Points")
    plt.colorbar()
    # Show plots
    plt.tight_layout()
    plt.show()
    print("hi")


def interploate_between():

    # Define 5 points in a 1x1 field
    x_points = np.array([[0.1, 0.1], [1, 0], [0, 1], [1, 1], [0.3, 0.3], [0.7, 0.7]])
    # Add random noise to x_points
    noise = np.random.normal(scale=0.01, size=x_points.shape)
    # x_points += noise
    y_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_points, dtype=torch.float64, device="cuda")
    y_train = torch.tensor(y_values, dtype=torch.float64, device="cuda")

    # Create the interpolator
    interpolator = InterpolatorND(x_train, y_train, num_cpu_cores=1)

    # Generate a grid of query points
    n_query = 10  # Number of query points along each axis
    x_query_vals = np.linspace(0, 1, n_query)
    y_query_vals = np.linspace(0, 1, n_query)
    x_query_grid, y_query_grid = np.meshgrid(x_query_vals, y_query_vals)
    x_query_flat = x_query_grid.flatten()
    y_query_flat = y_query_grid.flatten()

    # Convert query points to PyTorch tensors
    x_query = torch.tensor(
        np.column_stack((x_query_flat, y_query_flat)),
        dtype=torch.float64,
        device="cuda",
    )

    # Perform interpolation
    z_query = interpolator(x_query).cpu().numpy()  # Interpolated values

    # Plot the interpolated values
    plt.figure(figsize=(10, 10))
    plt.scatter(x_query_flat, y_query_flat, c=z_query, cmap="viridis", s=40)
    plt.scatter(x_points[:, 0], x_points[:, 1], c=y_values, cmap="viridis", s=200)
    plt.title("Interpolated Values")
    plt.colorbar()
    plt.show()
    print("hi")


if __name__ == "__main__":

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    interploate_between()

from typing import Literal

import numpy as np
import scipy
from stpy.helpers.voxel_grid import voxel_grid
from stpy.helpers.parallel_interpolation import InterpolatorArray
import torch

from stpy.borel_set import BorelSet
from stpy.embeddings.positive_embedding import PositiveEmbedding
from stpy.kernels import KernelFunction
from sklearn.decomposition import NMF
from nmf import run_nmf
from stpy.helpers.posterior_sampling import tmg
from fast_pytorch_kmeans import KMeans


class OptimalPositiveBasis(PositiveEmbedding):

    def __init__(
        self,
        *args,
        samples=300,
        discretization_size=30,
        data: torch.Tensor | BorelSet,
        fast_sampling=True,  # samples using squared gaussian instead of truncated gausian
        memory_limit=5,  # Limits the amount of points used for optimal basis construction
        sample_algorithm: Literal[
            "grid", "kmeans"
        ] = "grid",  # How to subsample if points are limited
        **kwargs,
    ):
        # roi is the set of points that the basis is optimal for if it is a tensor
        # else it is the region that the basis if optimal for that will be discretized
        # by discretization_size. If it is not given the entire domain will be used.
        super().__init__(*args, **kwargs)
        self.sample_algorithm = sample_algorithm
        self.num_samples = np.maximum(samples, self.m)
        self.fast = fast_sampling
        self.memory_limit = memory_limit if memory_limit is not None else 40
        self.interpolators = None

        if data is None:
            B = BorelSet(
                self.d,
                torch.tensor(
                    [[self.interval[0], self.interval[1]] for _ in range(self.d)]
                ).double(),
            )
            self.discretized_domain = B.return_discretization(discretization_size)
        elif isinstance(data, BorelSet):
            self.discretized_domain = data.return_discretization(discretization_size)
        else:
            self.discretized_domain = data

        y = self.discretized_domain[:, 0].view(-1, 1) * 0

        print("Optimal basis with arbitrary dimension, namely d =", self.d)
        print("Starting optimal basis construction, with m =", self.m)
        # self.new_kernel_object = KernelFunction(kernel_name=self.kernel_object.optkernel,
        # 										gamma = self.kernel_object.gamma, d = self.kernel_object.d)

        self.new_kernel_object = self.kernel_object
        self._fit_data(data=data)
        print("Optimal basis constructed.")
        if torch.sum(torch.isnan(self.embed_internal(self.discretized_domain))) > 0:
            print(
                "Failed basis? (zero is good):",
                torch.sum(torch.isnan(self.embed_internal(self.discretized_domain))),
            )
        self.precomp_integral = {}

    def get_m(self):
        return self.m

    def embed_internal(self, x):
        out = torch.zeros([len(x), self.m], dtype=torch.float64)
        for j in range(self.m):
            out[:, j] = self.basis_fun(x, j).view(-1)
        return out

    def basis_fun(self, x, j):
        raise Exception("Fit on data before using")

    def get_constraints(self):
        s = self.get_m()
        l = np.full(s, 0.0).astype(float)
        u = np.full(s, 10e10)
        Lambda = np.identity(s)
        return (l, Lambda, u)

    def integral(self, S):
        assert S.d == self.d

        if S in self.precomp_integral.keys():
            return self.precomp_integral[S]
        else:
            if S.d == 1:
                weights, nodes = S.return_legendre_discretization(n=256)
                psi = torch.sum(torch.diag(weights) @ self.embed_internal(nodes), dim=0)
                Gamma_half = self.cov()
                psi = Gamma_half.T @ psi
                self.precomp_integral[S] = psi
            elif S.d == 2:
                weights, nodes = S.return_legendre_discretization(n=50)
                vals = self.embed_internal(nodes)
                psi = torch.sum(torch.diag(weights) @ vals, dim=0)
                Gamma_half = self.cov()
                psi = Gamma_half.T @ psi
                self.precomp_integral[S] = psi
                if torch.sum(torch.isnan(psi)) > 0:
                    print("Failed integrals? (0 is good):", torch.sum(torch.isnan(psi)))

            else:
                raise NotImplementedError("Higher dimension not implemented.")
            return psi

    def cov(self, inverse=False):

        if self.precomp == False:

            x = self.discretized_domain
            vals = self.embed_internal(x)
            indices = torch.argmax(
                vals, dim=0
            )  # the nodes are the maxima of the bump functions
            t = x[indices]
            print("nodes of functions", t.size())

            self.Gamma = self.kernel(t, t)
            Z = self.embed_internal(t)

            M = torch.pinverse(Z.T @ Z + (self.s) * torch.eye(self.Gamma.size()[0]))
            self.M = torch.tensor(np.real(scipy.linalg.sqrtm(M.cpu().numpy())))

            self.Gamma_half = torch.tensor(
                np.real(
                    scipy.linalg.sqrtm(
                        self.Gamma.cpu().numpy()
                        + (self.s**2) * np.eye(self.Gamma.size()[0])
                    )
                )
            )
            self.Gamma_half = self.M @ self.Gamma_half
            self.invGamma_half = torch.pinverse(self.Gamma_half)
            self.precomp = True
        else:
            pass

        if inverse == True:
            return self.Gamma_half, self.invGamma_half
        else:
            return self.Gamma_half

    def _sample_gaussian_prior(self, x: torch.Tensor):
        n = self.num_samples
        dim = len(x)
        Cov = self.kernel_object.kernel(x, x) + 10e-7 * torch.eye(
            dim, dtype=torch.float64
        )
        L = torch.linalg.cholesky(Cov)
        if self.fast:
            random_vector = torch.normal(
                mean=torch.zeros(dim, n, dtype=torch.float64), std=1.0
            )
            y = torch.mm(L, random_vector) ** 2
        else:
            y = torch.tensor(
                tmg(
                    n,
                    np.zeros([dim], dtype=np.float64),
                    Cov.cpu().numpy(),
                    np.ones([dim], dtype=np.float64),
                    np.eye(dim, dtype=np.float64),
                    np.zeros(dim, dtype=np.float64),
                    verbose=True,
                ),
                dtype=torch.float64,
            )
        return y, L

    def _sample_gaussian_conditional(self, x_old, L_old, y_old, x):
        dim = len(x)  # dimensionality of input
        n = y_old.size(1)  # number of samples

        K_new_new = self.kernel_object.kernel(x, x) + 1e-7 * torch.eye(
            dim, dtype=torch.float64
        )
        K_new_old = self.kernel_object.kernel(x_old, x)

        alpha = torch.linalg.solve_triangular(L_old, y_old, upper=False)
        alpha = torch.linalg.solve_triangular(L_old.T, alpha, upper=True)

        mu_star = K_new_old @ alpha  # shape (dim, n)
        # TODO check if kernel is always symmetric
        K_old_new = K_new_old.T  # shape (dim_old, dim)

        tmp = torch.linalg.solve_triangular(L_old, K_old_new, upper=False)
        tmp2 = torch.linalg.solve_triangular(L_old.T, tmp, upper=True)

        Sigma_star = (
            K_new_new - (K_new_old @ tmp2) + 1e-7 * torch.eye(dim, dtype=torch.float64)
        )

        L_star = torch.linalg.cholesky(Sigma_star)
        if self.fast:
            random_vector_new = torch.normal(
                mean=torch.zeros(dim, n, dtype=torch.float64), std=1.0
            )
            y_new = (mu_star + L_star @ random_vector_new) ** 2
        else:
            y_new = torch.tensor(
                tmg(
                    n,
                    mu_star.cpu().numpy(),
                    Sigma_star.cpu().numpy(),
                    np.ones([dim], dtype=np.float64),
                    np.eye(dim, dtype=np.float64),
                    np.zeros(dim, dtype=np.float64),
                    verbose=True,
                ),
                dtype=torch.float64,
            )

        return y_new

    def _subsample_if_necessary(self, x: torch.Tensor):
        # Calculate number of clusters
        n_clusters = (self.memory_limit * 1_000_000_000) / x.element_size()
        # Since we want to calculate the cholesky decomp of the cov matrix of the data plus roi (expected to be 1% of data)
        n_clusters = int(np.sqrt(n_clusters) * 0.99 / 2.0)

        if len(x) > n_clusters:
            if self.sample_algorithm == "grid":
                centroids = voxel_grid(x, max_n_voxels=n_clusters)
                print(
                    f"Approximated data set with {len(centroids)} points for optimal"
                    " basis."
                )
                return centroids
            elif self.sample_algorithm == "kmeans":
                # Calculate maximum size of mini batch
                n_samples, n_features = x.shape
                SAFETY_FACTOR = 1.5
                max_batch_size = int(
                    (
                        self.memory_limit * 1_000_000_000
                        - 0.8 * n_samples
                        - 2 * n_clusters * n_features * x.element_size()
                    )
                    // (
                        (
                            n_features * n_clusters * x.element_size()
                            + n_features * x.element_size()
                        )
                        * SAFETY_FACTOR
                    )
                )
                if max_batch_size >= n_samples:
                    max_batch_size = None

                print(
                    f"Approximating data set with {n_clusters} points from"
                    f" {len(x)} points for optimal basis."
                    + (
                        f"Using batch size {max_batch_size}"
                        if max_batch_size is not None
                        else ""
                    )
                )
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    mode="euclidean",
                    verbose=1,
                    minibatch=max_batch_size,
                )
                kmeans.fit_predict(x)
                centroids = kmeans.centroids

                return centroids
        else:
            print("No subsampling necessary because data fits into memory")
            return x

    def _fit_data(self, data):
        self.data_m = self.m
        data = self._subsample_if_necessary(data)
        self.F_data, self.L_data = self._sample_gaussian_prior(data)
        self.F_data = self.F_data**2
        self.W_data, self.H_data, err = run_nmf(
            self.F_data,
            n_components=self.m,
            tol=1e-12,
            use_gpu=self.F_data.is_cuda,
            batch_max_iter=2000,
            fp_precision=self.F_data.dtype,
        )
        self.W_data = torch.tensor(self.W_data)
        self.H_data = torch.tensor(self.H_data)
        self.W_data = self.W_data / torch.linalg.norm(self.W_data, dim=0)
        self.data = data
        W_norm = self.W_data
        self._set_interpolators(data, W_norm)

    def basis_fun(self, q: torch.Tensor, j: int):
        if self.interpolators is None:
            raise Exception("Fit on data before using")

        return self.interpolators(j, q)

    def _set_interpolators(self, x: torch.Tensor, phi: torch.Tensor):
        assert x.dtype == phi.dtype
        self.interpolators = InterpolatorArray(x, phi, self.m)

    def fit(self, roi: torch.Tensor):
        assert self.data is not None, "Data must be given first"
        print("Refitting optimal basis")
        self.precomp = False
        x = torch.cat((self.data, roi), dim=0)
        F, _ = self._sample_gaussian_prior(x)
        F = F**2
        # Note: using cpu based NMF here since run_nmf has no way to pass initialization
        model = NMF(n_components=self.data_m, max_iter=200, tol=1e-8, init="custom")
        phi_roi_init = torch.zeros([len(roi), self.data_m], dtype=torch.float64)
        W_start = torch.cat((self.W_data, phi_roi_init), dim=0)
        W = torch.tensor(
            model.fit_transform(
                F.cpu().numpy(),
                W=W_start.cpu().numpy(),
                H=self.H_data.cpu().numpy(),
            )
        )
        self.Phi = W / torch.linalg.norm(W, dim=0)
        self.m = self.data_m
        self._set_interpolators(x, self.Phi)
        self.precomp = False
        self.precomp_integral = {}

    def add_new_functions(self, roi: torch.Tensor, n: int):
        x = torch.cat((self.data, roi), dim=0)
        F_new = self._sample_gaussian_conditional(
            self.data, self.L_data, self.F_data, roi
        )
        F = torch.cat([self.F_data, F_new])
        Phi_old = (
            torch.stack([self.basis_fun(x, j) for j in range(self.data_m)]).squeeze(2).T
        )
        Theta_old = self.H_data
        # TODO, theoretically this is wrong and we would have to solve over both Phi_old and Phi_new
        # also, caping at 0 has no theoretical underpinning
        objective = torch.clamp(F - Phi_old @ Theta_old, min=0)
        Phi_new, Theta_new, err = run_nmf(
            objective,
            n_components=n,
            tol=1e-7,
            use_gpu=True,
            batch_max_iter=100,
            fp_precision=objective.dtype,
        )
        Phi_new = torch.tensor(Phi_new)
        self.Phi = Phi_new / torch.linalg.norm(Phi_new, dim=0)
        self.m = self.data_m + n
        self.interpolators.set(1, x, self.Phi, n)
        self.precomp = False
        self.precomp_integral = {}


if __name__ == "__main__":

    from stpy.continuous_processes.gauss_procc import GaussianProcess
    from stpy.helpers.helper import interval
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    d = 2
    m = 5
    n = 64
    s = 0.01
    b = 0
    gamma = 0.5
    k = KernelFunction(gamma=gamma, d=2)

    xtest = torch.tensor(interval(n, d))

    xnew = xtest[:1000]

    xtest = xtest[1000:]

    Emb = OptimalPositiveBasis(
        d,
        m,
        offset=0.2,
        s=s,
        b=b,
        discretization_size=n,
        B=1000.0,
        kernel_object=k,
        data=xtest,
    )

    y, L = Emb._sample_prior(xtest, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    xx = xtest[:, 0].cpu().numpy()
    yy = xtest[:, 1].cpu().numpy()
    sc = ax.scatter(xx, yy, c=y.detach().numpy().reshape(-1), cmap="viridis")
    ax.grid(c="k", ls="-", alpha=0.1)
    plt.colorbar(sc)
    plt.title("Interpolated plot of y over xtest")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    ynew = Emb._sample_conditional(xtest, L, y, xnew)

    xtest = torch.cat([xtest, xnew])
    y = torch.cat([y, ynew])

    fig, ax = plt.subplots(figsize=(10, 6))
    xx = xtest[:, 0].cpu().numpy()
    yy = xtest[:, 1].cpu().numpy()
    sc = ax.scatter(xx, yy, c=y.detach().numpy().reshape(-1), cmap="viridis")
    ax.grid(c="k", ls="-", alpha=0.1)
    plt.colorbar(sc)
    plt.title("Interpolated plot of y over xtest")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    print("hi")

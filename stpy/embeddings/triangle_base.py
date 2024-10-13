import numpy as np
import scipy
import torch

from stpy.borel_set import BorelSet
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.embeddings.positive_embedding import PositiveEmbedding
from stpy.kernels import KernelFunction


class EfficientTriangleEmbedding(PositiveEmbedding):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._t = torch.linspace(
            self.interval[0], self.interval[1], steps=self.m, dtype=torch.float64
        )
        self._dm = (self.interval[1] - self.interval[0]) / (self.m - 1)

    def basis_fun(self, x: torch.Tensor, j: int):
        r"""
        Return the value of 1d basis function $\phi_{j}$
        over all dimensions of x

        :param x: double, need to be in the interval
        :param j: integer, index of hat functions, 0 <= j <= m-1
        :return: $\{\phi_j(x_1), \ldots, \phi_j(x_n)}$
        """
        res = torch.clamp(1 - torch.abs((x - self._t[j]) / self._dm), min=0)
        return res

    def integrate_1d(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor):
        """
        :param l: from
        :param u: to
        :param t: tensor of triangle centers
        :return: 1d integral over triangle basis functions given by centers and self.dm
        """

        def rising_integral(x):
            return (x - t + self._dm) ** 2 / (self._dm * 2.0)

        def falling_integral(x):
            return -((x - t - self._dm) ** 2) / (self._dm * 2.0)

        i = rising_integral(torch.clamp(b, t - self._dm, t)) - rising_integral(
            torch.clamp(a, t - self._dm, t)
        )
        i += falling_integral(torch.clamp(b, t, t + self._dm)) - falling_integral(
            torch.clamp(a, t, t + self._dm)
        )

        return i

    def integral(self, S):
        """
        Integrate the Phi(x) over S
        :param S: borel set
        :return: $\int_S \Phi(x) dx$
        """
        if S in self.procomp_integrals.keys():
            return self.procomp_integrals[S]

        else:
            assert S.d == self.d
            psi = torch.ones(self.m).double()
            if S.type == "box":
                psi = torch.tensor([1.0]).double()
                for i in range(self.d):
                    a, b = S.bounds[i, 0].double(), S.bounds[i, 1].double()
                    p = self.integrate_1d(a, b, self._t)
                    # multiply each with each element and flatten
                    psi = torch.outer(psi, p).flatten()

            elif S.type == "round":
                weights, nodes = S.return_legendre_discretization(30)
                vals = self.embed_internal(nodes)
                psi = weights.view(1, -1) @ vals

            Gamma_half = self.cov()
            emb = psi @ Gamma_half
            self.procomp_integrals[S] = emb
            return emb

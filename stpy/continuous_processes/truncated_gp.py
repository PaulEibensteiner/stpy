import numpy as np
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.helpers.posterior_sampling import tmg
import torch


class TruncatedGP:
    """
    A truncated Gaussian Process that can serve as a ground truth model
    for the PPP estimators. Sampling is very slow at the moment
    """

    def __init__(self, kernel, d):
        self.gp = GaussianProcess(kernel=kernel, d=d)
        self.x_acc = None
        self.y_acc = None

    def __call__(self, x: torch.tensor, dt: float = 1.0, burn_in=30):
        N = len(x)
        # Initialize sample array
        sample = torch.zeros(N)

        if self.x_acc is None:
            x_new = x
        else:
            # Find indices of x that are already in self.x_acc
            matching = torch.all(
                x.unsqueeze(1) == self.x_acc.unsqueeze(0), dim=2
            )  # (N, M)
            matching_indices = torch.nonzero(matching, as_tuple=False)  # (K, 2)
            idx_cached_in_x = matching_indices[:, 0]  # Indices in x
            idx_cached_in_acc = matching_indices[:, 1]  # Indices in self.x_acc

            # Determine which indices are new
            mask_cached = torch.zeros(N, dtype=torch.bool)
            mask_cached[idx_cached_in_x] = True
            idx_new = torch.nonzero(~mask_cached).squeeze(1)
            # Retrieve cached function values
            sample[idx_cached_in_x] = self.y_acc[idx_cached_in_acc]
            x_new = x[idx_new]

        # Compute function values for new points
        if len(x_new) > 0:
            if self.gp.fitted:
                mean_new, cov_new = self.gp.mean_std_sub(x_new, full=True)
                mean_new = mean_new.squeeze(1)
            else:
                mean_new = torch.zeros(
                    len(x_new),
                )
                cov_new = self.gp.kernel(
                    x,
                    x,
                )

            # Sample truncated GP for new points
            factor = torch.eye(len(x_new))
            summand = torch.zeros(len(x_new))
            cov_new = cov_new.cpu().numpy() + 1e-7 * np.eye(len(x_new))
            sample_new = tmg(
                1,
                mean_new.cpu().numpy(),
                cov_new,
                torch.ones(len(x_new)).cpu().numpy(),
                factor.cpu().numpy(),
                summand.cpu().numpy(),
                burn_in,
                True,
            )
            sample_new = torch.tensor(sample_new[0])

            # Update sample array and caches
            if self.x_acc is None:
                sample = sample_new
                self.x_acc = x_new
                self.y_acc = sample_new
            else:
                sample[idx_new] = sample_new
                self.x_acc = torch.cat([self.x_acc, x_new])
                self.y_acc = torch.cat([self.y_acc, sample_new])

            self.gp.fit(self.x_acc, self.y_acc.unsqueeze(1))

        return sample * dt

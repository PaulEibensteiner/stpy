import torch
from stpy.kernel_functions.kernel_params import KernelParams


def squared_exponential_kernel(a, b, **kwargs):
    """

    :param a:
    :param b:
    :param kwargs: must include gamma, kappa, group
    :return:
    """
    p = KernelParams(kwargs)
    p.assert_existence(["gamma", "kappa", "group"])

    a = a[:, p.group]
    b = b[:, p.group]
    # 	print (a.shape, b.shape)
    normx = torch.sum(a**2, dim=1).view(-1, 1)
    normy = torch.sum(b**2, dim=1).view(-1, 1)

    product = torch.mm(b, torch.t(a))
    # sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
    sqdist = -2 * product + torch.t(normx) + normy
    arg = (-0.5 / (p.gamma * p.gamma)) * sqdist
    res = torch.exp(arg)
    return p.kappa * res


def squared_exponential_kernel_diag(a, b, **kwargs):
    p = KernelParams(kwargs)
    p.assert_existence(["gamma", "kappa", "group"])

    a = a[:, p.group]
    b = b[:, p.group]
    sqdist = (a - b) ** 2
    arg = (-0.5 / (p.gamma * p.gamma)) * sqdist
    res = torch.exp(arg)
    return p.kappa * res


def squared_exponential_integral(a_x, a_y, b_x, b_y, **kwargs):
    """
    Returns a function that computes g(x) for multiple 2D points x given lower and upper bounds.

    Parameters:
    - a_x: torch.Tensor, lower bounds in x-dimension (shape: [N])
    - a_y: torch.Tensor, lower bounds in y-dimension (shape: [N])
    - b_x: torch.Tensor, upper bounds in x-dimension (shape: [N])
    - b_y: torch.Tensor, upper bounds in y-dimension (shape: [N])
    - kwargs: should give attributes gamma (float) and kappa (float)

    Returns:
    - A function `g(x)` that computes g(x) for input x (torch.Tensor of shape [M, 2]).
    """
    p = KernelParams(kwargs)
    p.assert_existence(["gamma", "kappa"])
    gamma = p.gamma
    kappa = p.kappa

    def g(x):
        """
        Compute the integral g(x) for multiple 2D points x.

        Parameters:
        - x: torch.Tensor, input points of shape [M, 2] where each row is a 2D point.

        Returns:
        - torch.Tensor of shape [len(a_x), len(x)], where result[i][j] is g(x_j) for bounds from a_x[i], a_y[i], b_x[i], b_y[i].
        """
        x1, x2 = x[:, 0], x[:, 1]  # Extract x1 and x2 from input tensor x
        a_x_broadcast = a_x.unsqueeze(1)  # Shape [N, 1]
        a_y_broadcast = a_y.unsqueeze(1)  # Shape [N, 1]
        b_x_broadcast = b_x.unsqueeze(1)  # Shape [N, 1]
        b_y_broadcast = b_y.unsqueeze(1)  # Shape [N, 1]

        # Compute the error function terms
        erf_x1_a = torch.erf((a_x_broadcast - x1) * torch.sqrt(torch.tensor(gamma)))
        erf_x1_b = torch.erf((b_x_broadcast - x1) * torch.sqrt(torch.tensor(gamma)))
        erf_x2_a = torch.erf((a_y_broadcast - x2) * torch.sqrt(torch.tensor(gamma)))
        erf_x2_b = torch.erf((b_y_broadcast - x2) * torch.sqrt(torch.tensor(gamma)))

        # Compute the product of error function differences
        integral_values = (erf_x1_a - erf_x1_b) * (erf_x2_a - erf_x2_b)

        # Scale by constants
        result = (torch.pi * kappa / (4 * gamma)) * integral_values

        return result

    return g


if __name__ == "__main__":
    # Test squared_exponential_integral
    a_x = torch.tensor([-float("inf"), -float("inf")])
    a_y = torch.tensor([-float("inf"), -float("inf")])
    b_x = torch.tensor([float("inf"), float("inf")])
    b_y = torch.tensor([float("inf"), float("inf")])

    gamma = 1.0
    kappa = 1.0
    g = squared_exponential_integral(a_x, a_y, b_x, b_y, gamma=gamma, kappa=kappa)
    x = torch.tensor([[87, 0], [1123, 11]])
    assert torch.allclose(g(x), torch.tensor([torch.pi, torch.pi]))

    # Test with new bounds x in [0,1] and y in [0,1]
    a_x = torch.tensor([0.0])
    a_y = torch.tensor([0.0])
    b_x = torch.tensor([1.0])
    b_y = torch.tensor([1.0])

    g = squared_exponential_integral(a_x, a_y, b_x, b_y, gamma=10e-6, kappa=kappa)
    x = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    assert torch.allclose(g(x), torch.tensor([1.0, 1.0]))

    a_x = torch.tensor([0.0, 1.0])
    a_y = torch.tensor([0.0, 2.0])
    b_x = torch.tensor([1.0, 3.0])
    b_y = torch.tensor([1.0, 4.0])

    g = squared_exponential_integral(a_x, a_y, b_x, b_y, gamma=0.5, kappa=3.0)
    x = torch.tensor([[0.5, 0.5], [2.0, 3.0]])
    result = g(x)
    assert torch.allclose(
        result, torch.tensor([[2.7639, 0.0548], [0.3794, 8.7851]]), atol=1e-4
    )

    torch.ones(())

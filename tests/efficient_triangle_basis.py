from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.embeddings.bump_bases import TriangleEmbedding
from stpy.embeddings.triangle_base import EfficientTriangleEmbedding
import torch


if __name__ == "__main__":
    d = 1
    m = 2
    S = BorelSet(1, torch.tensor([[-1, 1]]))

    inefficient = TriangleEmbedding(d=d, m=m, interval=(-1, 1))
    efficient = EfficientTriangleEmbedding(d, m, interval=(-1, 1))

    for x, j in [(0.5, 1), (0.1, 0)]:
        x = torch.tensor(x)
        assert inefficient.basis_fun(x, j) == efficient.basis_fun(x, j)

    for set in [[-1, 1], [-0.25, 1], [-2, 2]]:
        s = BorelSet(1, torch.tensor([set]))
        assert torch.all(inefficient.integral(s) == efficient.integral(s))

    d = 2
    m = 2

    inefficient = TriangleEmbedding(d=d, m=m, interval=(-1, 1))
    efficient = EfficientTriangleEmbedding(d, m, interval=(-1, 1))

    for x, j in [([0.5, 0.1], 1), ([0.7, 0.1], 0)]:
        x = torch.tensor(x)
        assert torch.all(inefficient.basis_fun(x, j) == efficient.basis_fun(x, j))

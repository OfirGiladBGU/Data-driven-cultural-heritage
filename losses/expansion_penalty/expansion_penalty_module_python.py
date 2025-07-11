import time
import torch
from torch.autograd import Function
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


class expansionPenaltyFunction(Function):
    @staticmethod
    def forward(ctx, xyz, primitive_size, alpha):
        batchsize, n, _ = xyz.size()
        assert n % primitive_size == 0
        num_elements = n // primitive_size

        xyz_np = xyz.detach().cpu().numpy()
        dist = np.zeros((batchsize, n), dtype=np.float32)
        father = -np.ones((batchsize, n), dtype=np.int32)
        mean_mst_length = np.zeros((batchsize,), dtype=np.float32)

        for b in range(batchsize):
            for e in range(num_elements):
                start = e * primitive_size
                end = (e + 1) * primitive_size
                points = xyz_np[b, start:end, :]

                dists = cdist(points, points, metric='euclidean')
                mst = minimum_spanning_tree(dists).toarray()
                mst = mst + mst.T

                edge_indices = np.argwhere(mst > 0)
                edge_lengths = mst[mst > 0]
                mean_length = edge_lengths.mean()
                mean_mst_length[b] += mean_length

                for i, j in edge_indices:
                    if dists[i, j] > alpha * mean_length:
                        dist[b, start + i] = dists[i, j]
                        father[b, start + i] = start + j

        mean_mst_length /= num_elements
        ctx.save_for_backward(xyz, torch.tensor(father, device=xyz.device))

        return (
            torch.tensor(dist, device=xyz.device),
            torch.tensor(father, device=xyz.device),
            torch.tensor(mean_mst_length, device=xyz.device)
        )

    @staticmethod
    def backward(ctx, grad_dist, grad_father, grad_mml):
        xyz, father = ctx.saved_tensors
        batchsize, n, _ = xyz.size()
        grad_xyz = torch.zeros_like(xyz)

        for b in range(batchsize):
            for i in range(n):
                j = father[b, i].item()
                if j >= 0:
                    delta = xyz[b, i] - xyz[b, j]
                    norm = torch.norm(delta) + 1e-6
                    grad_xyz[b, i] += grad_dist[b, i] * delta / norm

        return grad_xyz, None, None


class expansionPenaltyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, primitive_size, alpha):
        return expansionPenaltyFunction.apply(input, primitive_size, alpha)


def test_expansion_penalty():
    x = torch.rand(2, 1024, 3).cuda()
    print("Input_size:", x.shape)
    expansion = expansionPenaltyModule()
    start_time = time.perf_counter()
    dist, father, mean_length = expansion(x, 128, 1.5)
    print("Runtime: %.6fs" % (time.perf_counter() - start_time))


# test_expansion_penalty()

import torch
from torch.autograd import Function
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


class expansionPenaltyFunction(Function):
    @staticmethod
    def forward(ctx, xyz, primitive_size, alpha):
        B, N, _ = xyz.shape
        assert N % primitive_size == 0, "Points must be divisible by primitive_size"
        num_elements = N // primitive_size

        xyz_np = xyz.detach().cpu().numpy()
        dist_out = np.zeros((B, N), dtype=np.float32)
        assignment_out = -np.ones((B, N), dtype=np.int32)
        mean_mst_lengths = np.zeros((B,), dtype=np.float32)

        for b in range(B):
            for e in range(num_elements):
                start = e * primitive_size
                end = (e + 1) * primitive_size
                points = xyz_np[b, start:end, :]  # [primitive_size, 3]

                # Compute pairwise distances
                dists = cdist(points, points, metric='euclidean')

                # Compute MST
                mst = minimum_spanning_tree(dists).toarray()
                mst = mst + mst.T  # make undirected

                edge_indices = np.argwhere(mst > 0)
                edge_lengths = mst[mst > 0]
                mean_length = edge_lengths.mean()
                mean_mst_lengths[b] += mean_length

                for i, j in edge_indices:
                    if dists[i, j] > alpha * mean_length:
                        dist_out[b, start + i] = dists[i, j]
                        assignment_out[b, start + i] = start + j

        mean_mst_lengths /= num_elements
        ctx.save_for_backward(xyz, torch.tensor(assignment_out, device=xyz.device))

        return (
            torch.tensor(dist_out, device=xyz.device),
            torch.tensor(assignment_out, device=xyz.device),
            torch.tensor(mean_mst_lengths, device=xyz.device)
        )

    @staticmethod
    def backward(ctx, grad_dist, grad_assignment, grad_mml):
        xyz, assignment = ctx.saved_tensors
        B, N, _ = xyz.shape
        grad_xyz = torch.zeros_like(xyz)

        for b in range(B):
            for i in range(N):
                j = assignment[b, i].item()
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

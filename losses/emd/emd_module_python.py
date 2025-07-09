import torch
from torch import nn
from torch.autograd import Function


class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        B, N, _ = xyz1.shape
        dist = torch.zeros(B, N, device=xyz1.device)
        assignment = torch.zeros(B, N, dtype=torch.long, device=xyz1.device)

        for b in range(B):
            p1 = xyz1[b]  # [N, 3]
            p2 = xyz2[b].clone()  # [N, 3]
            used = torch.zeros(N, dtype=torch.bool, device=xyz1.device)
            for i in range(N):
                dists = ((p1[i] - p2) ** 2).sum(-1)  # [N]
                dists[used] = float('inf')
                idx = torch.argmin(dists)
                dist[b, i] = dists[idx]
                assignment[b, i] = idx
                used[idx] = True

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        B, N, _ = xyz1.shape
        grad_xyz1 = torch.zeros_like(xyz1)

        for b in range(B):
            for i in range(N):
                j = assignment[b, i].item()
                delta = xyz1[b, i] - xyz2[b, j]
                norm = torch.norm(delta) + 1e-6
                grad_xyz1[b, i] += grad_dist[b, i] * delta / norm

        return grad_xyz1, None, None, None


class emdModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2, eps=0.005, iters=50):
        return emdFunction.apply(xyz1, xyz2, eps, iters)

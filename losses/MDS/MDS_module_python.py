import torch
import numpy as np
from torch import nn
from torch.autograd import Function


def minimum_density_sampling(xyz, npoint, mean_mst_length):
    """
    Inputs:
        xyz: (B, N, 3) float32
        npoint: int
        mean_mst_length: (B,) float32
    Returns:
        idx: (B, npoint) int64
    """
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)

    for b in range(B):
        cur_xyz = xyz[b]  # [N, 3]
        selected = []
        distances = torch.ones(N, device=xyz.device) * 1e10

        # randomly select the first point
        farthest = torch.randint(0, N, (1,), device=xyz.device).item()
        for i in range(npoint):
            selected.append(farthest)
            idx[b, i] = farthest
            centroid = cur_xyz[farthest].unsqueeze(0)  # [1, 3]
            dist = torch.norm(cur_xyz - centroid, dim=1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()

    return idx


class MinimumDensitySamplingFunction(Function):
    @staticmethod
    def forward(ctx, xyz, npoint, mean_mst_length):
        idx = minimum_density_sampling(xyz, npoint, mean_mst_length)
        return idx

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None


minimum_density_sample = MinimumDensitySamplingFunction.apply


class GatherOperationFunction(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        features: [B, C, N]
        idx: [B, npoint]
        returns: [B, C, npoint]
        """
        B, C, N = features.shape
        _, S = idx.shape

        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, npoint]
        output = torch.gather(features, 2, idx_expanded)
        ctx.save_for_backward(idx, torch.tensor(N))
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.saved_tensors
        B, C, S = grad_out.shape
        grad_features = torch.zeros(B, C, N.item(), device=grad_out.device)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
        grad_features.scatter_add_(2, idx_expanded, grad_out)
        return grad_features, None


gather_operation = GatherOperationFunction.apply

import torch
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
    batchsize, n, _ = xyz.shape
    idx = torch.zeros(batchsize, npoint, dtype=torch.long, device=xyz.device)

    for b in range(batchsize):
        cur_xyz = xyz[b]  # [n, 3]
        selected = []
        distances = torch.ones(n, device=xyz.device) * 1e10
        threshold = mean_mst_length[b].item() * 0.5  # Example usage

        # randomly select the first point
        farthest = torch.randint(0, n, (1,), device=xyz.device).item()
        for i in range(npoint):
            selected.append(farthest)
            idx[b, i] = farthest
            centroid = cur_xyz[farthest].unsqueeze(0)  # [1, 3]
            dist = torch.norm(cur_xyz - centroid, dim=1)

            # apply mean_mst_length threshold to prevent selecting points too close
            dist[dist < threshold] = 1e10

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
        batchsize, channels, n = features.shape
        _, npoint = idx.shape

        idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1)  # [B, C, npoint]
        output = torch.gather(features, 2, idx_expanded)
        ctx.save_for_backward(idx, torch.tensor(n))
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, n = ctx.saved_tensors
        batchsize, channels, npoint = grad_out.shape
        grad_features = torch.zeros(batchsize, channels, n.item(), device=grad_out.device)
        idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1)
        grad_features.scatter_add_(2, idx_expanded, grad_out)
        return grad_features, None


gather_operation = GatherOperationFunction.apply

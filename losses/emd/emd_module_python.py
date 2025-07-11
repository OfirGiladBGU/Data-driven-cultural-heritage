import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Function


class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        batchsize, n, _ = xyz1.size()
        dist = torch.zeros(batchsize, n, device=xyz1.device)
        assignment = torch.full((batchsize, n), -1, dtype=torch.int32, device=xyz1.device)

        for b in range(batchsize):
            p1 = xyz1[b]  # [n, 3]
            p2 = xyz2[b].clone()  # [n, 3]
            used = torch.zeros(n, dtype=torch.bool, device=xyz1.device)
            for i in range(n):
                dists = ((p1[i] - p2) ** 2).sum(-1)  # [n]
                dists[used] = float('inf')
                idx = torch.argmin(dists)
                dist[b, i] = dists[idx]
                assignment[b, i] = idx
                used[idx] = True

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, grad_dist, grad_assignment):
        xyz1, xyz2, assignment = ctx.saved_tensors
        batchsize, n, _ = xyz1.size()
        grad_xyz1 = torch.zeros_like(xyz1)

        for b in range(batchsize):
            for i in range(n):
                j = assignment[b, i].item()
                if j >= 0:
                    delta = xyz1[b, i] - xyz2[b, j]
                    norm = torch.norm(delta) + 1e-6
                    grad_xyz1[b, i] += grad_dist[b, i] * delta / norm

        return grad_xyz1, None, None, None


class emdModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2, eps=0.005, iters=50):
        return emdFunction.apply(xyz1, xyz2, eps, iters)


def test_emd():
    x1 = torch.rand(20, 8192, 3).cuda()
    x2 = torch.rand(20, 8192, 3).cuda()
    emd = emdModule()
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2, 0.05, 3000)
    print("Input_size: ", x1.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis = 1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())


# test_emd()

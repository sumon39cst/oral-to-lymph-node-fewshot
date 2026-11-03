"""Pure-PyTorch selective state-space scan.

The official `mamba_ssm` package implements this recurrence as a CUDA kernel that only
builds on Linux. This module computes exactly the same recurrence with ordinary tensor
operations, so MambaVision (and its pretrained weights) run unchanged on Windows.
It is slower than the kernel but numerically equivalent; it is only put on the import
path when the real package is not installed.

Discretised state-space model, per channel d and state n, over sequence positions l:
    h_l = exp(delta_l * A) * h_{l-1} + delta_l * B_l * u_l
    y_l = <h_l, C_l> + D * u_l
"""
import torch
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """u, delta: [b, d, l]; A: [d, n]; B, C: [b, n, l] (input-dependent) or [d, n]; D: [d]."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    A, B, C = A.float(), B.float(), C.float()
    b, d, length = u.shape
    n = A.shape[1]

    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if B.dim() == 3:
        delta_bu = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        delta_bu = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)

    # One fused multiply-add per position keeps the Python loop cheap; the read-out with C
    # is done once for the whole sequence afterwards.
    h = u.new_zeros((b, d, n))
    states = []
    for a_i, bu_i in zip(delta_a.unbind(2), delta_bu.unbind(2)):
        h = torch.addcmul(bu_i, a_i, h)
        states.append(h)
    states = torch.stack(states, dim=2)                         # [b, d, l, n]
    if C.dim() == 3:
        y = torch.einsum("bdln,bnl->bdl", states, C)
    else:
        y = torch.einsum("bdln,dn->bdl", states, C)

    if D is not None:
        y = y + u * D.float().view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype_in)
    return (y, h) if return_last_state else y

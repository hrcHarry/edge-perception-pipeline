# src/robustness/kalman.py
import torch

class ScalarKF:
    def __init__(self, q=1e-3, r=1e-2, x0=None, p0=1.0, device="cpu"):
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        self.r = torch.as_tensor(r, dtype=torch.float32, device=device)
        self.x = None if x0 is None else torch.as_tensor(x0, dtype=torch.float32, device=device)
        self.p = torch.as_tensor(p0, dtype=torch.float32, device=device)

    def step(self, z):
        # predict
        if self.x is None:
            self.x = z.clone()
        self.p = self.p + self.q
        # update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x

def kf_smooth_probs(prob_seq, q=1e-3, r=1e-2):
    device = prob_seq.device
    T, C = prob_seq.shape
    xs = []
    kfs = [ScalarKF(q=q, r=r, x0=prob_seq[0, c], device=device) for c in range(C)]
    for t in range(T):
        zt = prob_seq[t]  # (C,)
        xt = torch.empty_like(zt)
        for c in range(C):
            xt[c] = kfs[c].step(zt[c])
        xt = torch.clamp(xt, 0, 1)
        xt = xt / (xt.sum() + 1e-12)
        xs.append(xt)
    return torch.stack(xs, dim=0)


import torch
import torch.nn as nn

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if not self.training or torch.rand(1) > self.p:
            return x

        B, C, H, W = x.size()
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.std(dim=[2, 3], keepdim=True)

        mu_shuffle = mu[torch.randperm(B)]
        sigma_shuffle = sigma[torch.randperm(B)]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample([B, 1, 1, 1]).to(x.device)
        mu_mixed = mu * lam + mu_shuffle * (1 - lam)
        sigma_mixed = sigma * lam + sigma_shuffle * (1 - lam)

        return sigma_mixed * (x - mu) / (sigma + 1e-6) + mu_mixed

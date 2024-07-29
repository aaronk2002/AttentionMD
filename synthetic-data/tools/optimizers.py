import torch
from torch.optim import Optimizer

eps = torch.finfo(torch.double).eps


class SMD(Optimizer):
    def __init__(self, params, p, normalize=False):
        defaults = dict(p=p)
        self.p = p
        self.normalize = normalize
        super().__init__(params, defaults)

    def step(self):
        for weight in self.param_groups:
            for param in weight["params"]:
                with torch.no_grad():
                    if not self.normalize:
                        param_prime = (
                            torch.sign(param) * torch.abs(param) ** (self.p - 1.0)
                            - weight["lr"] * param.grad
                        )
                        param.copy_(
                            torch.sign(param_prime)
                            * torch.abs(param_prime) ** (1 / (self.p - 1.0))
                        )
                    else:
                        param_prime = torch.sign(param) * torch.abs(param) ** (
                            self.p - 1.0
                        ) - weight["lr"] * param.grad / (
                            param.grad.norm(self.p / (self.p - 1)) + eps
                        )
                        param.copy_(
                            torch.sign(param_prime)
                            * torch.abs(param_prime) ** (1 / (self.p - 1.0))
                        )

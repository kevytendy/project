# dp_optimizer.py

import torch
from torch.optim import Optimizer


class DPSGD(Optimizer):
    def __init__(self, params, lr, noise_multiplier, max_grad_norm, clip_decay_rate=1.0, **kwargs):
        super(DPSGD, self).__init__(params, **kwargs)
        self.lr = lr
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clip_decay_rate = clip_decay_rate  # Add decay rate for dynamic clipping
        self.epoch = 0  # Track current epoch

    def step(self, closure=None):
        # Update clipping bound based on epoch
        current_clip_bound = self.max_grad_norm * (self.clip_decay_rate ** self.epoch)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(p, max_norm=current_clip_bound)
                # Add noise for DP
                p.grad.add_(torch.randn_like(p.grad) * self.noise_multiplier * current_clip_bound)

        self.epoch += 1  # Increment epoch counter
        super(DPSGD, self).step(closure)
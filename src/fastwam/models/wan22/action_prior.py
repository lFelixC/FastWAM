import math

import torch
from torch import nn


class ActionPriorHead(nn.Module):
    """Low-frequency DCT action prior conditioned on current proprioception."""

    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        num_freq: int,
        hidden: int = 256,
        num_layers: int = 2,
        prior_noise_scale: float = 0.0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.num_freq = int(num_freq)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.prior_noise_scale = float(prior_noise_scale)

        if self.cond_dim <= 0:
            raise ValueError(f"`cond_dim` must be > 0, got {self.cond_dim}.")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be > 0, got {self.action_dim}.")
        if self.num_freq <= 0:
            raise ValueError(f"`num_freq` must be > 0, got {self.num_freq}.")
        if self.hidden <= 0:
            raise ValueError(f"`hidden` must be > 0, got {self.hidden}.")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}.")

        layers: list[nn.Module] = [nn.Linear(self.cond_dim, self.hidden), nn.SiLU()]
        for _ in range(self.num_layers - 1):
            layers.extend([nn.Linear(self.hidden, self.hidden), nn.SiLU()])
        out = nn.Linear(self.hidden, self.num_freq * self.action_dim)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.mlp = nn.Sequential(*layers)

        self.register_buffer("_dct_analysis", torch.empty(0), persistent=False)
        self.register_buffer("_dct_synthesis", torch.empty(0), persistent=False)
        self._dct_horizon: int | None = None

    def _build_basis(self, horizon: int, *, device: torch.device) -> None:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError(f"`horizon` must be > 0, got {horizon}.")
        if self.num_freq > horizon:
            raise ValueError(f"`num_freq` ({self.num_freq}) must be <= horizon ({horizon}).")

        dtype = torch.float32
        t = torch.arange(horizon, device=device, dtype=dtype).unsqueeze(0)
        k = torch.arange(horizon, device=device, dtype=dtype).unsqueeze(1)
        dmat = torch.cos(math.pi * (2.0 * t + 1.0) * k / (2.0 * float(horizon)))
        alpha = torch.full((horizon, 1), math.sqrt(2.0 / float(horizon)), device=device, dtype=dtype)
        alpha[0] = math.sqrt(1.0 / float(horizon))
        dmat = dmat * alpha

        analysis = dmat[: self.num_freq].contiguous()
        synthesis = analysis.transpose(0, 1).contiguous()
        self._dct_analysis = analysis
        self._dct_synthesis = synthesis
        self._dct_horizon = horizon

    def _ensure_basis(self, horizon: int, *, device: torch.device) -> None:
        if (
            self._dct_horizon != int(horizon)
            or self._dct_analysis.numel() == 0
            or self._dct_analysis.device != device
        ):
            self._build_basis(horizon, device=device)

    def forward(self, cond: torch.Tensor, horizon: int) -> torch.Tensor:
        if cond.ndim != 2:
            raise ValueError(f"`cond` must be 2D [B, D], got shape {tuple(cond.shape)}.")
        if cond.shape[1] != self.cond_dim:
            raise ValueError(f"`cond` last dim must be {self.cond_dim}, got {cond.shape[1]}.")
        self._ensure_basis(int(horizon), device=cond.device)
        coeff = self.mlp(cond).view(cond.shape[0], self.num_freq, self.action_dim)
        prior = torch.einsum("tk,bkd->btd", self._dct_synthesis, coeff.float())
        return prior.to(dtype=coeff.dtype)

    def lowpass_target(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3:
            raise ValueError(f"`action` must be 3D [B, T, D], got shape {tuple(action.shape)}.")
        if action.shape[2] != self.action_dim:
            raise ValueError(f"`action` last dim must be {self.action_dim}, got {action.shape[2]}.")
        horizon = int(action.shape[1])
        self._ensure_basis(horizon, device=action.device)
        coeff = torch.einsum("kt,btd->bkd", self._dct_analysis, action.float())
        lowpass = torch.einsum("tk,bkd->btd", self._dct_synthesis, coeff)
        return lowpass.detach().to(dtype=action.dtype)

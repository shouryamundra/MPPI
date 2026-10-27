"""Holonomic MPPI for the IGHA* opendubs planner: the planner's dynamics kernel, a path-tracking cost, and the controller.

State    x, y, yaw (world), vx, vy, omega (body), as in the planner.
Control  u = (ax, ay, alpha) in [-1, 1]; physical acceleration = u * accel_max, alpha_max.

    cfg = yaml.safe_load(open("mppi/Configs/Holonomic_config.yaml"))
    mppi = HolonomicMPPI(ighastar_root, node_info, cfg)    # node_info: the planner's node_info (limits, footprint, dt)
    mppi.set_costmap(costmap, center_xy)                   # 255 free .. 0 lethal, rows = y, cols = x
    vx, vy, omega = mppi.step(state6, ref)                 # ref: T x 6 reference states, one per dt
    mppi.predict(state6)                                   # T x 6, the plan rolled out from state6
"""
from pathlib import Path

import numpy as np
import torch

from mppi.MPPI import MPPI


class OpendubsDynamics(torch.nn.Module):
    """The planner's opendubs_kernel on a flat map, launched for all K rollouts at once (mppi/cuda/opendubs_rollout.cu)."""

    def __init__(self, ighastar_root, accel_max, alpha_max, dt, device):
        super().__init__()
        from torch.utils.cpp_extension import load
        env = Path(ighastar_root) / "ighastar" / "src" / "Environments"
        self.kernel = load(name="mppi_opendubs_rollout", sources=[str(Path(__file__).parent / "cuda" / "opendubs_rollout.cu")],
                           extra_include_paths=[str(env / "src"), str(env / "include")], extra_cuda_cflags=["-O3"])
        self.dt, self.accel_max, self.alpha_max = float(dt), float(accel_max), float(alpha_max)
        self.elev = torch.zeros((2, 2), dtype=torch.float32, device=device)

    def forward(self, states, controls):
        """states M x K x T x 6 (every step a copy of the start state), controls M x K x T x 3 -> M x K x T x 6."""
        M, K, T, _ = controls.shape
        s0 = states[:, :, 0].reshape(M * K, 6).contiguous().clone()
        u = controls.reshape(M * K, T, 3).contiguous()
        out = torch.empty((M * K, T, 6), dtype=torch.float32, device=u.device)
        valid = torch.empty(M * K, dtype=torch.bool, device=u.device)
        cost = torch.empty(M * K, dtype=torch.float32, device=u.device)
        self.kernel.rollout(s0, u, out, valid, cost, self.elev, self.dt, self.accel_max, self.alpha_max)
        return out.view(M, K, T, 6)


class AccelSampling(torch.nn.Module):
    """Gaussian noise on the normalized controls. The weights use the temperature that gives an effective sample size of ess * K."""

    def __init__(self, noise, temperature, ess, mppi_cfg, device):
        super().__init__()
        self.d, self.ess, self.temperature = device, float(ess), float(temperature)
        self.sigma = torch.as_tensor(noise, dtype=torch.float32, device=device)
        self.nu = self.sigma.numel()
        self.K, self.T = int(mppi_cfg["ROLLOUTS"]), int(mppi_cfg["TIMESTEPS"])
        self.noise = torch.zeros((self.K, self.T, self.nu), device=device)

    def reset(self):
        torch.manual_seed(0)

    def sample(self, state, U):
        perturbed = (U + torch.randn((self.K, self.T, self.nu), device=self.d) * self.sigma).clamp(-1.0, 1.0)
        self.noise = perturbed - U                                          # the perturbation actually applied
        return perturbed.unsqueeze(0), self.temperature * (U * self.noise / self.sigma ** 2).sum((1, 2))

    def weights(self, c):
        """Softmax of -c / temperature, with the smallest temperature (bisection in log space) that has ESS >= ess * K."""
        lo, hi = torch.log(c.max() * 1e-4 + 1e-6), torch.log(c.max() + 1e-6)
        for _ in range(25):
            mid = (lo + hi) / 2
            small = 1.0 / (torch.softmax(-c / mid.exp(), 0) ** 2).sum() < self.ess * c.numel()
            lo, hi = torch.where(small, mid, lo), torch.where(small, hi, mid)
        return torch.softmax(-c / ((lo + hi) / 2).exp(), 0)

    def update_control(self, cost_total, U, state):
        w = self.weights(cost_total.reshape(-1) - cost_total.min())
        U = (U + (w.view(-1, 1, 1) * self.noise).sum(0)).clamp(-1.0, 1.0)
        return U.unsqueeze(0), U


class TrackingCost(torch.nn.Module):
    """Squared error to the reference, costmap lethality at the footprint, and the planner's velocity limits as penalties."""

    def __init__(self, cost_cfg, map_cfg, node_info, device):
        super().__init__()
        f = dict(device=device, dtype=torch.float32)
        self.w = {k: float(cost_cfg[k + "_w"]) for k in ("pos", "heading", "vel", "end", "lethal", "limit")}
        n = node_info
        self.vx_min, self.vx_max, self.omega_lim = float(n["min_vel"]), float(n["max_vel"]), float(n["max_omega"])
        self.vy_max, self.vy_floor, self.k_strafe = float(n["vy_max"]), float(n["vy_floor"]), float(n["K_strafe"])
        hl, hw = float(n["length"]) / 2, float(n["width"]) / 2
        self.corners = torch.tensor([[0, 0], [hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]], **f)
        self.res = float(map_cfg["map_res"])
        self.n_px = round(float(map_cfg["map_size"]) / self.res)
        self.costmap = torch.full((self.n_px, self.n_px), 255.0, **f)
        self.center = torch.zeros(2, **f)
        self.ref = torch.zeros((1, 6), **f)

    def set_costmap(self, costmap, center_xy):
        self.costmap = torch.as_tensor(costmap, device=self.corners.device, dtype=torch.float32)
        self.center = torch.as_tensor(center_xy, device=self.corners.device, dtype=torch.float32)

    def lethal(self, pose):
        c, s = pose[..., 2:3].cos(), pose[..., 2:3].sin()
        x = pose[..., 0:1] + c * self.corners[:, 0] - s * self.corners[:, 1] - self.center[0]
        y = pose[..., 1:2] + s * self.corners[:, 0] + c * self.corners[:, 1] - self.center[1]
        ix = (x / self.res + self.n_px / 2).long().clamp(0, self.n_px - 1)
        iy = (y / self.res + self.n_px / 2).long().clamp(0, self.n_px - 1)
        return ((255.0 - self.costmap[iy, ix]) / 255.0).max(-1).values

    def forward(self, states, controls):
        err = states - self.ref
        err[..., 2] = torch.atan2(err[..., 2].sin(), err[..., 2].cos())
        vx, vy, om = states[..., 3], states[..., 4], states[..., 5]
        vy_cap = (self.vy_max - self.k_strafe * vx.abs()).clamp(min=self.vy_floor)
        over = ((vx - self.vx_max).clamp(min=0) + (self.vx_min - vx).clamp(min=0) + (vy.abs() - vy_cap).clamp(min=0)
                + (om.abs() - self.omega_lim).clamp(min=0))
        step = (self.w["pos"] * err[..., :2].square().sum(-1) + self.w["heading"] * err[..., 2].square()
                + self.w["vel"] * err[..., 3:].square().sum(-1) + self.w["lethal"] * self.lethal(states[..., :3])
                + self.w["limit"] * over.square())
        return step.sum(-1) + self.w["end"] * step[..., -1]                  # M x K


class HolonomicMPPI:
    def __init__(self, ighastar_root, node_info, cfg, device="cuda"):
        self.d = device
        mppi_cfg = {"ROLLOUTS": int(cfg["MPPI_config"]["ROLLOUTS"]), "TIMESTEPS": int(cfg["MPPI_config"]["TIMESTEPS"]),
                    "BINS": 1, "u_per_command": 1}
        s = cfg["Sampling_config"]
        self.dynamics = OpendubsDynamics(ighastar_root, node_info["accel_max"], node_info["alpha_max"], node_info["dt"], device)
        self.cost = TrackingCost(cfg["Cost_config"], cfg["Map_config"], node_info, device)
        self.mppi = MPPI(self.dynamics, self.cost, AccelSampling(s["noise"], s["temperature"], s["ess"], mppi_cfg, device),
                         mppi_cfg, device)
        self.reset()

    def reset(self) -> None:
        self.mppi.reset()

    def set_costmap(self, costmap, center_xy) -> None:
        self.cost.set_costmap(costmap, center_xy)

    @torch.no_grad()
    def predict(self, state) -> np.ndarray:
        s0 = torch.as_tensor(state, device=self.d, dtype=torch.float32).view(1, 1, 1, 6)
        T = self.mppi.T
        return self.dynamics(s0.expand(1, 1, T, 6), self.mppi.U.view(1, 1, T, 3))[0, 0].cpu().numpy()

    @torch.no_grad()
    def step(self, state, ref) -> np.ndarray:
        """The body velocity after the first control, as the dynamics say it is reached."""
        s = torch.as_tensor(state, device=self.d, dtype=torch.float32)
        self.cost.ref = torch.as_tensor(ref, device=self.d, dtype=torch.float32)
        u0 = self.mppi(s)[0, 0]
        return self.dynamics(s.view(1, 1, 1, 6), u0.view(1, 1, 1, 3))[0, 0, 0, 3:].cpu().numpy()

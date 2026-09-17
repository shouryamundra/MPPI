#!/usr/bin/env python3
"""Analytical bicycle dynamics + BEV (CUDA). Extracted from BeamNGRL UW_mppi."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _load_bicycle_kernel():
    dyn_dir = Path(__file__).resolve().parent
    cpp = str(dyn_dir / "analytical_bicycle.cpp")
    cu = str(dyn_dir / "analytical_bicycle.cu")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    try:
        from hound_nav.cuda_cache import load_extension

        return load_extension(
            "mppi_analytical_bicycle",
            [cpp, cu],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except ImportError:
        from torch.utils.cpp_extension import load

        return load(
            name="mppi_analytical_bicycle",
            sources=[cpp, cu],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _load_bicycle_kernel()
    return _KERNEL


class SimpleCarDynamics:
    """CUDA bicycle rollout over a height/normal BEV."""

    def __init__(
        self,
        Dynamics_config,
        Map_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        self.dtype = dtype
        self.d = device

        # Controls are physical: steer [rad], wheelspeed [m/s]. Kernel still
        # multiplies by these scales — keep identity (1.0).
        self.throttle_to_wheelspeed = np.float32(
            Dynamics_config.get("throttle_to_wheelspeed", 1.0)
        )
        self.steering_max = np.float32(Dynamics_config.get("steering_max", 1.0))

        self.dt_default = np.float32(Dynamics_config["dt"])
        self.dt = self.dt_default
        self.K = np.int32(MPPI_config["ROLLOUTS"])
        self.T = np.int32(MPPI_config["TIMESTEPS"])
        self.M = np.int32(MPPI_config["BINS"])
        # Plant size: bicycle kernel uses indices 0..14; last NC are controls.
        # Default 17 matches BeamNG / HOUND control_state layout.
        nx = int(
            Dynamics_config.get(
                "NX",
                Dynamics_config.get("state_dims", Dynamics_config.get("control_state_dims", 17)),
            )
        )
        if nx < 17:
            raise ValueError(
                f"SimpleCarDynamics NX must be >= 17 (got {nx}); "
                "CUDA bicycle layout uses state[0..14] + 2 control channels"
            )
        self.NX = np.int32(nx)
        self.NC = np.int32(int(Dynamics_config.get("NC", 2)))

        self.BEVmap_size = np.float32(Map_config["map_size"])
        self.BEVmap_res = np.float32(Map_config["map_res"])
        self.BEVmap_size_px = np.int32(self.BEVmap_size / self.BEVmap_res)

        self.states = torch.zeros(
            (self.M, self.K, self.T, self.NX), dtype=self.dtype, device=self.d
        )

        self.D = np.float32(Dynamics_config["D"])
        self.B = np.float32(Dynamics_config["B"])
        self.C = np.float32(Dynamics_config["C"])
        self.lf = np.float32(Dynamics_config["lf"])
        self.lr = np.float32(Dynamics_config["lr"])
        self.Iz = np.float32(Dynamics_config["Iz"])
        self.LPF_tau = np.float32(Dynamics_config["LPF_tau"])
        self.LPF_st = np.float32(Dynamics_config["LPF_st"])
        self.LPF_th = np.float32(Dynamics_config["LPF_th"])
        self.res_coeff = np.float32(Dynamics_config["res_coeff"])
        self.drag_coeff = np.float32(Dynamics_config["drag_coeff"])

        self.car_l2 = np.float32(Dynamics_config["car_length"] / 2)
        self.car_w2 = np.float32(Dynamics_config["car_width"] / 2)
        self.cg_height = np.float32(Dynamics_config["cg_height"])

        self.block_dim = 32
        self.grid_dim = int(np.ceil(self.K / self.block_dim))

        kernel = _kernel()
        dyn_type = Dynamics_config["type"]
        if dyn_type == "slip3d":
            self.model_forward = kernel.rollout_slip3d
        elif dyn_type == "noslip3d":
            self.model_forward = kernel.rollout_noslip3d
        else:
            raise ValueError(f"Unknown Dynamics_config type={dyn_type!r}")

        self.BEVmap_height = torch.zeros(
            (self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d
        )
        self.BEVmap_normal = torch.zeros(
            (3, self.BEVmap_size_px, self.BEVmap_size_px),
            dtype=self.dtype,
            device=self.d,
        )

    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal

    def get_states(self):
        return self.states

    def forward(self, state, controls):
        state_ = state.squeeze(0)
        controls_ = controls.squeeze(0)
        self.model_forward(
            state_,
            controls_,
            self.BEVmap_height,
            self.BEVmap_normal,
            self.dt,
            self.K,
            self.T,
            self.NX,
            self.NC,
            self.D,
            self.B,
            self.C,
            self.lf,
            self.lr,
            self.Iz,
            self.throttle_to_wheelspeed,
            self.steering_max,
            self.BEVmap_size_px,
            self.BEVmap_res,
            self.BEVmap_size,
            self.car_l2,
            self.car_w2,
            self.cg_height,
            self.LPF_tau,
            self.LPF_st,
            self.LPF_th,
            self.res_coeff,
            self.drag_coeff,
            self.block_dim,
            self.grid_dim,
        )
        self.states = torch.clone(state_).unsqueeze(0)
        return self.states

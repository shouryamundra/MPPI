# mppi

Standalone **MPPI** control library extracted from `BeamNGRL.control.UW_mppi`.

## Contents

- `mppi.MPPI` — Williams-style MPPI shell
- `mppi.Sampling.Delta_Sampling` — delta-control sampling (trailing `control_dim` state channels)
- `mppi.Dynamics.SimpleCarDynamicsTCUDA` — CUDA bicycle + BEV (`analytical_bicycle.{cu,cpp}` next to the module)
- `mppi.Costs` — **example** pure-torch costs (not used by HOUND deployment)
  - `SimpleCarCost` — goal-seeking (`set_goal` / `set_BEV`)
  - `TrackingCost` — path tracking (`set_path` / `set_BEV`)
- `mppi/Configs/*.yaml` — sample Dynamics / Sampling / Cost / Map / MPPI configs

HOUND deployment tracking cost is `hound_nav.trackingCostCUDA.SimpleCarCost` (CUDA).
Swap that in the same `MPPI(Dynamics, Costs, Sampling, …)` constructor.

## Cost API (for custom costs)

```text
forward(states, controls) -> [K]     # M×K×T×NX states, M×K×T×nu controls
set_BEV(height, normal, cost)        # optional
set_goal(xy) / set_path(T×4)         # optional
```

## Example

```bash
# after colcon build + source install/setup.bash
python3 /path/to/mppi/examples/wire_mppi_cost.py --cost tracking
python3 /path/to/mppi/examples/wire_mppi_cost.py --cost goal
```

## NX / plant size

Pass `Dynamics_config["NX"]` (or `state_dims` / `control_state_dims`). Default
and minimum is **17** (indices 0..14 used by the bicycle kernel; last 2 = actuators).

## Build

```bash
cd ~/colcon_ws && colcon build --packages-select mppi hound_nav --symlink-install
```

## Holonomic MPPI (wip-holonomic)

`mppi/holonomic.py` is MPPI for a holonomic robot with the IGHA* opendubs planner's dynamics: state `(x, y, yaw, vx, vy, omega)`
(yaw in the world frame, velocities in the body frame), normalized acceleration controls `(ax, ay, alpha)` in [-1, 1].
It holds the dynamics (`mppi/cuda/opendubs_rollout.cu`, a torch binding that includes IGHAStar's `opendubs.cu`), the sampling,
the tracking cost and `HolonomicMPPI`. The weights are in `mppi/Configs/Holonomic_config.yaml`.

```python
cfg = yaml.safe_load(open("mppi/Configs/Holonomic_config.yaml"))
mppi = HolonomicMPPI(ighastar_root, node_info, cfg)   # node_info: the planner's node_info (limits, footprint, dt)
mppi.set_costmap(costmap, center_xy)                  # optional, 255 free .. 0 lethal
vx, vy, omega = mppi.step(state6, ref)                # ref: T x 6 reference states, one per dt
```

`python examples/holonomic_line.py ighastar_root planner.yml` tracks a straight line from 0.2 m off it (needs CUDA, IGHAStar and
the planner's yaml). The config was tuned on that line only.

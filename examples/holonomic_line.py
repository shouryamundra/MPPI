"""Straight-line check for mppi/holonomic.py with mppi/Configs/Holonomic_config.yaml: starts 0.2 m off a 6 m line (0.4 m/s) and a
12 m line (1.0 m/s), prints the RMS distance to the line after 4 s. python examples/holonomic_line.py [ighastar_root planner_yaml]"""
import math, os, sys, yaml, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mppi.holonomic import HolonomicMPPI
root = sys.argv[1] if len(sys.argv) > 1 else "/root/colcon_ws/od_sim/deps/IGHAStar"
ni = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "/root/colcon_ws/od_sim/config/opendubs_planner_v2.yml"))["experiment_info_default"]["node_info"]
cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "mppi", "Configs", "Holonomic_config.yaml")))
dt, T = ni["dt"], cfg["MPPI_config"]["TIMESTEPS"]
for v, length in ((0.4, 6.0), (1.0, 12.0)):
    m = HolonomicMPPI(root, ni, cfg); s = [0.0, 0.2, 0, 0, 0, 0]; cte = []
    for i in range(int(length / (v * dt))):
        ref = [[(i + 1 + t) * v * dt, 0.0, 0.0, v, 0.0, 0.0] for t in range(T)]
        m.step(s, np.array(ref, dtype=np.float32)); s = m.predict(s)[0].tolist(); cte.append(abs(s[1]))
    print(f"line at {v} m/s: rms after 4 s {np.sqrt(np.mean(np.square(cte[40:]))):.3f} m, max {max(cte):.3f} m, final {cte[-1]:.3f} m")

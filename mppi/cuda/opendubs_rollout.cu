// Torch binding for IGHAStar's opendubs_kernel: K rollouts of T steps on tensors already on the GPU.
// opendubs.cu is included from IGHAStar (put its Environments/src and Environments/include on the include path), not copied.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "opendubs.cu"

// state K x 6 (advanced in place), controls K x T x 3 (normalized), out K x T x 6, valid K (bool), cost K, flat 2D zero map.
void rollout(torch::Tensor state, torch::Tensor controls, torch::Tensor out, torch::Tensor valid, torch::Tensor cost,
             torch::Tensor elev, double dt, double accel_max, double alpha_max) {
  const int K = state.size(0), T = controls.size(1);
  TerrainMap map;                         // flat: elevation 0 everywhere, so roll = pitch = 0 inside the kernel
  map.type = MAP_ELEVATION;
  map.elev = elev.data_ptr<float>();
  map.cost = elev.data_ptr<float>();
  map.nx = elev.size(1); map.ny = elev.size(0);
  map.voxel_xy = 1.0f; map.res_inv = 1.0f;
  const float big = 1e9f;                 // the planner's rejection limits are off; the MPPI cost penalizes them
  const int threads = 128, blocks = (K + threads - 1) / threads;
  opendubs_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(), out.data_ptr<float>(), controls.data_ptr<float>(), map, valid.data_ptr<bool>(),
      cost.data_ptr<float>(), (float)dt, T, K, 6, 3, 0.35f, 0.35f, (float)accel_max, (float)alpha_max,
      big, -big, big, big, big, big, big, big, 0.0f);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rollout", &rollout); }

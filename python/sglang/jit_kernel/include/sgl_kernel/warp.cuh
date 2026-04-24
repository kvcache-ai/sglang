#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>

// Some warp primitives
namespace device::warp {

static constexpr uint32_t kFullMask = 0xffffffffu;

template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
    value = value + __shfl_xor_sync(active_mask, value, mask, 32);
  return value;
}

template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_max(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
    value = math::max(value, __shfl_xor_sync(active_mask, value, mask, 32));
  return value;
}

}  // namespace device::warp

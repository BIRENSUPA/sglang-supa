/// SUPA DLPack device compatibility for upstream CUDA JIT sources.
///
/// Torch-SUPA is a PyTorch PrivateUse1 backend and exports device tensors as
/// kDLExtDev.  Upstream CUDA kernels constrain those tensors as kDLCUDA.  This
/// header is force-included only in SUPA JIT translation units, where it makes
/// that pair compare as equivalent without changing either enum value.

#pragma once

#include <concepts>
#include <dlpack/dlpack.h>

inline constexpr bool operator==(DLDeviceType lhs, DLDeviceType rhs) {
  const auto lhs_value = static_cast<int32_t>(lhs);
  const auto rhs_value = static_cast<int32_t>(rhs);
  return lhs_value == rhs_value ||
         ((lhs_value == kDLCUDA && rhs_value == kDLExtDev) ||
          (lhs_value == kDLExtDev && rhs_value == kDLCUDA));
}

inline constexpr bool operator!=(DLDeviceType lhs, DLDeviceType rhs) {
  return !(lhs == rhs);
}

// gstub nvcc does not support --expt-relaxed-constexpr for functions with
// explicit __device__ annotation. Upstream utils.cuh declares div_ceil as
// __device__ constexpr, which NVIDIA nvcc allows from host code but gstub
// does not. Provide a __host__ overload so host-side callers resolve here.
namespace sglang {
namespace device {
template <std::integral T, std::integral U>
__host__ inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}
} // namespace device
} // namespace sglang

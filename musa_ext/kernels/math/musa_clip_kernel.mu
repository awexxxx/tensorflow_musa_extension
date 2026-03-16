#include <musa_fp16.h>
#include <musa_runtime.h>

#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

template <typename T>
struct ClipComputeType {
  using type = T;
};

template <>
struct ClipComputeType<Eigen::half> {
  using type = float;
};

template <>
struct ClipComputeType<bfloat16> {
  using type = float;
};

template <typename T>
__device__ __forceinline__ T DeviceMin(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__device__ __forceinline__ T DeviceMax(T a, T b) {
  return a > b ? a : b;
}

__device__ __forceinline__ float LoadValue(const float* p) { return *p; }
__device__ __forceinline__ double LoadValue(const double* p) { return *p; }
__device__ __forceinline__ int32 LoadValue(const int32* p) { return *p; }
__device__ __forceinline__ int64 LoadValue(const int64* p) { return *p; }

__device__ __forceinline__ float LoadValue(const Eigen::half* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float LoadValue(const bfloat16* p) {
  float out = 0.0f;
  uint32_t* dst = reinterpret_cast<uint32_t*>(&out);
  const uint16_t* src = reinterpret_cast<const uint16_t*>(p);
  *dst = static_cast<uint32_t>(*src) << 16;
  return out;
}

__device__ __forceinline__ void StoreValue(float v, float* p) { *p = v; }
__device__ __forceinline__ void StoreValue(double v, double* p) { *p = v; }
__device__ __forceinline__ void StoreValue(int32 v, int32* p) { *p = v; }
__device__ __forceinline__ void StoreValue(int64 v, int64* p) { *p = v; }

__device__ __forceinline__ void StoreValue(float v, Eigen::half* p) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}

__device__ __forceinline__ void StoreValue(float v, bfloat16* p) {
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&v);
  *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>((*src) >> 16);
}

template <typename T>
__global__ void ClipKernel(const T* x, const T* lo, const T* hi, bool lo_scalar,
                           bool hi_scalar, T* out, int n) {
  using ComputeT = typename ClipComputeType<T>::type;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  const int lo_idx = lo_scalar ? 0 : idx;
  const int hi_idx = hi_scalar ? 0 : idx;
  ComputeT x_val = LoadValue(&x[idx]);
  ComputeT lo_val = LoadValue(&lo[lo_idx]);
  ComputeT hi_val = LoadValue(&hi[hi_idx]);

  ComputeT clipped = DeviceMin(DeviceMax(x_val, lo_val), hi_val);
  StoreValue(clipped, &out[idx]);
}

template <typename T>
void LaunchClipKernel(const T* x, const T* lo, const T* hi, bool lo_scalar,
                      bool hi_scalar, T* out, int n, musaStream_t stream) {
  if (n <= 0) return;

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  ClipKernel<T><<<blocks, threads, 0, stream>>>(x, lo, hi, lo_scalar,
                                                 hi_scalar, out, n);
}

template void LaunchClipKernel<float>(const float*, const float*, const float*,
                                      bool, bool, float*, int, musaStream_t);
template void LaunchClipKernel<double>(const double*, const double*, const double*,
                                       bool, bool, double*, int, musaStream_t);
template void LaunchClipKernel<int32>(const int32*, const int32*, const int32*,
                                      bool, bool, int32*, int, musaStream_t);
template void LaunchClipKernel<int64>(const int64*, const int64*, const int64*,
                                      bool, bool, int64*, int, musaStream_t);
template void LaunchClipKernel<Eigen::half>(
    const Eigen::half*, const Eigen::half*, const Eigen::half*, bool, bool,
    Eigen::half*, int, musaStream_t);
template void LaunchClipKernel<bfloat16>(const bfloat16*, const bfloat16*,
                                         const bfloat16*, bool, bool,
                                         bfloat16*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow

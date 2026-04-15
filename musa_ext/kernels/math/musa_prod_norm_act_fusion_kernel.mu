#include <math.h>
#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

using bfloat16 = tensorflow::bfloat16;

namespace {

constexpr int kActTanh = 0;
constexpr int kActSigmoid = 1;
constexpr int kActLog = 2;

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFloat(float* p, float v) { *p = v; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float result = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&result);
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return result;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  const uint32_t* f_ptr = reinterpret_cast<const uint32_t*>(&v);
  uint16_t b_val = static_cast<uint16_t>((*f_ptr) >> 16);
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

__device__ __forceinline__ float ApplyActivation(float x, int activation_kind) {
  switch (activation_kind) {
    case kActSigmoid:
      return 1.0f / (1.0f + expf(-x));
    case kActLog:
      return logf(x);
    case kActTanh:
    default:
      return tanhf(x);
  }
}

template <typename T>
__global__ void ProdNormActKernel(const T* src, T* dst, int64_t outer_dim,
                                  int64_t inner_size, float epsilon,
                                  int activation_kind) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (idx >= inner_size) {
    return;
  }

  float prod = 1.0f;
  for (int64_t outer = 0; outer < outer_dim; ++outer) {
    const float value = LoadFloat(src + outer * inner_size + idx);
    prod *= value * value;
  }

  const float clamped = fmaxf(prod, epsilon);
  StoreFloat(dst + idx, ApplyActivation(clamped, activation_kind));
}

}  // namespace

template <typename T>
void LaunchProdNormAct(const T* src, T* dst, int64_t outer_dim,
                       int64_t inner_size, float epsilon, int activation_kind,
                       musaStream_t stream) {
  if (inner_size <= 0) {
    return;
  }

  const int block_size = 256;
  const int64_t grid_size = (inner_size + block_size - 1) / block_size;
  ProdNormActKernel<<<grid_size, block_size, 0, stream>>>(
      src, dst, outer_dim, inner_size, epsilon, activation_kind);
}

template void LaunchProdNormAct<float>(const float*, float*, int64_t, int64_t,
                                       float, int, musaStream_t);
template void LaunchProdNormAct<Eigen::half>(const Eigen::half*,
                                             Eigen::half*, int64_t, int64_t,
                                             float, int, musaStream_t);
template void LaunchProdNormAct<bfloat16>(const bfloat16*, bfloat16*, int64_t,
                                          int64_t, float, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow

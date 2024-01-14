#include <cstdint>

__global__ void sum_atomic(const float *inp, float *dest, int64_t size)
{
  const int64_t i = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (i >= size)
  {
    return;
  }
  atomicAdd(dest, inp[i]);
}
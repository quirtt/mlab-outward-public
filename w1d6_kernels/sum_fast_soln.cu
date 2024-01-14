__global__ void sum_1024(const float *inp, float *dest, int size)
{
    static __shared__ float data[1024];
    int tidx = threadIdx.x;
    int idx = tidx + blockIdx.x * 1024;
    data[tidx] = (idx < size) ? inp[idx] : 0;

    for (int chunk_size = 1024 / 2; chunk_size > 0; chunk_size /= 2)
    {
        __syncthreads();
        if (tidx < chunk_size)
        {
            data[tidx] += data[tidx + chunk_size];
        }
    }

    if (tidx == 0)
    {
        dest[blockIdx.x] = data[tidx];
    }
}
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "reduce_sum.cuh"


__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp32(const float4 * __restrict   src, 
                                                float* __restrict           dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v1, 
                                                const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::float4_sum(_recv._vf);

    //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);
    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}



__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_int32(const int4 * __restrict   src, 
                                                int32_t* __restrict           dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v1, 
                                                const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(0);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::int4_sum(_recv._vi);

    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp16_fp32(const float4 * __restrict   src, 
                                                float* __restrict           dst,
                                                const uint32_t              Wsrc_v8, 
                                                uint32_t                    Wdst_v1, 
                                                const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrs[i] = 0;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::half8_sum(_recv._vf);

    //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);
    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_u8_i32(const int4 * __restrict     src, 
                                                  int32_t* __restrict         dst,
                                                  const uint32_t              Wsrc_v16, 
                                                  uint32_t                    Wdst_v1, 
                                                  const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v16 = decx::utils::ceil<uint32_t>(proc_dims.x, 16);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v16 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        /*
        * Because of the vec-load, some don't care values will be loaded
        * at the end of the thread grid (at the end of the process area as well)
        * For reduced summation process, set the don't care value(s) to all zero
        * to eliminate their effect
        */
        if (tidx == proc_W_v16 - 1) {
            uint32_t _left_u8 = proc_W_v16 * 16 - proc_dims.x;

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(0, tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    _thread_sum = decx::reduce::GPUK::uchar16_sum(_recv._vi);

    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp32_transp(const float4 * __restrict   src, 
                                                       float* __restrict           dst,
                                                       const uint32_t              Wsrc_v4, 
                                                       uint32_t                    Wdst_v1, 
                                                       const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::float4_sum(_recv._vf);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}



__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_int32_transp(const int4 * __restrict    src, 
                                                       int32_t* __restrict         dst,
                                                       const uint32_t              Wsrc_v4, 
                                                       uint32_t                    Wdst_v1, 
                                                       const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(0);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::int4_sum(_recv._vi);

    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}





__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp16_fp32_transp(const float4 * __restrict   src, 
                                                            float* __restrict           dst,
                                                            const uint32_t              Wsrc_v8, 
                                                            uint32_t                    Wdst_v1, 
                                                            const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrs[i] = 0;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::half8_sum(_recv._vf);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_h_u8_i32_transp(const int4 * __restrict     src, 
                                                         int32_t* __restrict         dst,
                                                         const uint32_t              Wsrc_v16, 
                                                         uint32_t                    Wdst_v1, 
                                                         const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;

    uint32_t proc_W_v16 = decx::utils::ceil<uint32_t>(proc_dims.x, 16);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v16 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        
        if (tidx == proc_W_v16 - 1) {
            uint32_t _left_u8 = proc_W_v16 * 16 - proc_dims.x;

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(0, tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    _thread_sum = decx::reduce::GPUK::uchar16_sum(_recv._vi);

    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp32(const float4 * __restrict   src, 
                                                float4* __restrict          dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v4, 
                                                const uint2                 proc_dims_v4)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v4 + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float2 tmp1, tmp2, tmp3, tmp4;
    
    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly 
    * where the matrix ends.
    */
    if (tidx < proc_dims_v4.x && tidy < proc_dims_v4.y) {
        _recv._vf = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp1 = ((float2*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2 = ((float2*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__float_add2, &tmp1, &tmp3);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__float_add2, &tmp2, &tmp4);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((float2*)_workspace[0])[threadIdx.y * 4 + threadIdx.x / 8] = tmp3;
        ((float2*)_workspace[0])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4;
    }

    __syncthreads();
    
    _recv._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < proc_dims_v4.x && threadIdx.y == 0) {
        dst[STG_dex] = _recv._vf;
    }
}





__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_v_int32(const int4 * __restrict   src, 
                                                 int4* __restrict          dst,
                                                 const uint32_t              Wsrc_v4, 
                                                 uint32_t                    Wdst_v4, 
                                                 const uint2                 proc_dims_v4)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v4 + tidx;

    __shared__ int4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    int2 tmp1, tmp2, tmp3, tmp4;
    
    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly 
    * where the matrix ends.
    */
    if (tidx < proc_dims_v4.x && tidy < proc_dims_v4.y) {
        _recv._vi = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vi;

    __syncthreads();

    tmp1 = ((int2*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2 = ((int2*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__i32_add2, &tmp1, &tmp3);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__i32_add2, &tmp2, &tmp4);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((int2*)_workspace[0])[threadIdx.y * 4 + threadIdx.x / 8] = tmp3;
        ((int2*)_workspace[0])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4;
    }

    __syncthreads();
    
    _recv._vi = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < proc_dims_v4.x && threadIdx.y == 0) {
        dst[STG_dex] = _recv._vi;
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp16_fp32(const float4 * __restrict   src, 
                                                     float4* __restrict          dst,
                                                     const uint32_t              Wsrc_v8, 
                                                     uint32_t                    Wdst_v4, 
                                                     const uint2                 proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex_x = threadIdx.x + blockDim.x * blockIdx.x * 2 + threadIdx.y * blockDim.x;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2;
    decx::utils::_cuda_vec64 tmp3, tmp4;
    
    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly 
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv._vf = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp1._vd.x = ((double*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2._vd.x = ((double*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp1._arrh2[0], &tmp3._v_h2_2[0]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp1._arrh2[1], &tmp3._v_h2_2[1]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp2._arrh2[0], &tmp4._v_h2_2[0]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp2._arrh2[1], &tmp4._v_h2_2[1]);

    tmp1._arrf2[0] = __half22float2(tmp3._v_h2_2[0]);
    tmp1._arrf2[1] = __half22float2(tmp3._v_h2_2[1]);
    tmp2._arrf2[0] = __half22float2(tmp4._v_h2_2[0]);
    tmp2._arrf2[1] = __half22float2(tmp4._v_h2_2[1]);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        _workspace[0][threadIdx.y * 4 + threadIdx.x / 8] = tmp1._vf;
        _workspace[1][threadIdx.y * 4 + threadIdx.x / 8] = tmp2._vf;
    }

    __syncthreads();
    
    if (STG_dex_x < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && threadIdx.y < 2) {
        tmp1._vf = _workspace[threadIdx.y][threadIdx.x];
        dst[blockIdx.y * Wdst_v4 + STG_dex_x] = tmp1._vf;
    }
#endif
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_v_u8_i32(const int4 * __restrict   src, 
                                                  int4* __restrict          dst,
                                                  const uint32_t            Wsrc_v16, 
                                                  uint32_t                  Wdst_v4, 
                                                  const uint2               proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
    uint64_t STG_dex_x = threadIdx.x + blockDim.x * blockIdx.x * 4 + threadIdx.y * blockDim.x;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;
    
    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly 
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 16) && tidy < proc_dims_v1.y) {
        _recv._vi = src[LDG_dex];
    }

    /*for (int i = 0; i < 16; ++i) {
        _recv._arru8[i] = 2;
    }*/
    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    // checker load 4x4 uint8 values
    tmp1._vui.x = ((uint32_t*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2._vui.x = ((uint32_t*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];
    tmp3._vui.x = ((uint32_t*)_workspace[threadIdx.x % 8])[64 + threadIdx.y * 4 + threadIdx.x / 8];
    tmp4._vui.x = ((uint32_t*)_workspace[threadIdx.x % 8])[96 + threadIdx.y * 4 + threadIdx.x / 8];

    // expand each uint8 from 8-bit to 16-bit
    tmp1._vd.x = decx::utils::cuda::__cvt_uchar4_ushort4(tmp1._vi.x);
    tmp2._vd.x = decx::utils::cuda::__cvt_uchar4_ushort4(tmp2._vi.x);
    tmp3._vd.x = decx::utils::cuda::__cvt_uchar4_ushort4(tmp3._vi.x);
    tmp4._vd.x = decx::utils::cuda::__cvt_uchar4_ushort4(tmp4._vi.x);

    __syncwarp(0xffffffff);

    // execute warp-level 16-bit x2 reduce procedure on 16 x 8<reduced>, and store the results in tmp[1, 4].y
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__u16_add4, &tmp1._vd.x, &tmp1._vd.y);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__u16_add4, &tmp2._vd.x, &tmp2._vd.y);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__u16_add4, &tmp3._vd.x, &tmp3._vd.y);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__u16_add4, &tmp4._vd.x, &tmp4._vd.y);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        _recv._vi = make_int4(tmp1._arrs[4], tmp1._arrs[5], tmp1._arrs[6], tmp1._arrs[7]);
        _workspace[0][threadIdx.y * 4 + threadIdx.x / 8] = _recv._vf;

        _recv._vi = make_int4(tmp2._arrs[4], tmp2._arrs[5], tmp2._arrs[6], tmp2._arrs[7]);
        _workspace[1][threadIdx.y * 4 + threadIdx.x / 8] = _recv._vf;

        _recv._vi = make_int4(tmp3._arrs[4], tmp3._arrs[5], tmp3._arrs[6], tmp3._arrs[7]);
        _workspace[2][threadIdx.y * 4 + threadIdx.x / 8] = _recv._vf;

        _recv._vi = make_int4(tmp4._arrs[4], tmp4._arrs[5], tmp4._arrs[6], tmp4._arrs[7]);
        _workspace[3][threadIdx.y * 4 + threadIdx.x / 8] = _recv._vf;
    }

    __syncthreads();
    
    if (STG_dex_x < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && threadIdx.y < 4) {
        tmp1._vf = _workspace[threadIdx.y][threadIdx.x];
        dst[blockIdx.y * Wdst_v4 + STG_dex_x] = tmp1._vi;
    }
}
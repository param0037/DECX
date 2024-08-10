/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

/**
* Kernel naming conventions:
* cu_GEMM_<data_type>_<proc_type>_kernel_<W>_<H>_<T>
* All the paramters are in v1.
* 
*                       W
*                _______________
*               |               |
*               |      B        | L
*               |               |
*               |               |
*      L         ---------------
*  _________     _______________
* |         |   |               |
* |         |   |               |
* |         |   |               | 
* |    A    |   |      dst      | H
* |         |   |               |
* |         |   |               |
*  ---------     ---------------
*
* data_type : fp32, fp64, fp16, cplxf, cplxd
* proc_type : S for simplified matrix multiplication (dst = AB)
*             F for full GEMM form (dst = alpha*AB + beta*C)
* T : T is optional, if labled, means this function is operating on
*       the transposed form of matrix A. pitchA_v1 parameter should be
*       the pitch of the transposed matrix A.
*
* Loading a transposed form of matrix A, helps reduce the load sectors
* per memory request when loading matrix A, expecially when template
* variable L is small.
*/

#ifndef _GEMM_KERNELS_CUH_
#define _GEMM_KERNELS_CUH_

#include "../../../../../common/basic.h"
#include "../../../../../common/CUSV/decx_cuda_vectypes_ops.cuh"
#include "../../../../../common/decx_utils_functions.h"


namespace decx
{
namespace blas
{
    // FP32
namespace GPUK
{
    // thread = dim3(32, 8)
    // reg_per_thread = 114 (32-bit), shared_per_block = 16640 Byte
    // mio_compute_ratio = 1.5625e-2
    __global__ void cu_GEMM_fp32_kernel_16_128_128(const float* __restrict A, const float* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp32_F_kernel_16_128_128(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst,
        const float alpha, const float beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    // thread = dim3(32, 8)
    // reg_pe_thread = 114 (32-bit), shared_per_block = 33280 Byte
    // mio_compute_ratio = 1.5625e-2
    __global__ void cu_GEMM_fp32_kernel_32_128_128(const float* __restrict A, const float* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp32_F_kernel_32_128_128(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst,
        const float alpha, const float beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    // thread = dim3(32, 8)
    // L = 32 : reg_per_thread = 127 (32-bit); shared_per_block = 32768 Byte
    // L = 16 : reg_per_thread = 123 (32-bit); shared_per_block = 16384 Byte
    // L = 8 : reg_per_thread = 125 (32-bit); shared_per_block = 8192 Byte
    // mio_compute_ratio = 1.5625e-2
    template<uint32_t L>
    __global__ void cu_GEMM_fp32_kernel_128_128_T(const float* __restrict A, const float* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    template<uint32_t L>
    __global__ void cu_GEMM_fp32_F_kernel_128_128_T(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst,
        const float alpha, const float beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    /**
    * thread = [16, 16]; proc_dims = [64, 64], thread_loc = [4, 4]
    * reg_per_thread = 64 (32-bit); shared_per_block = 16896 Byte
    * mio_compute_ratio = 3.1250e-2
    */
    __global__ void cu_GEMM_fp32_kernel_32_64_64(const float* __restrict A, const float* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp32_F_kernel_32_64_64(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst,
        const float alpha, const float beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    // reg_per_thread = 64 (32-bit); shared_per_block = 8448
    // mio_compute_ratio = 3.1250e-2
    __global__ void cu_GEMM_fp32_kernel_16_64_64(const float* __restrict A, const float* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp32_F_kernel_16_64_64(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst,
        const float alpha, const float beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    /**
     * L = 32 : reg_per_thread = 63 (32-bit); shared_per_block = 16384 Byte
     * L = 16 : reg_per_thread = 63 (32-bit); shared_per_block = 8192 Byte
     * mio_compute_ratio = 3.1250e-2
    */
    template<uint32_t L>
    __global__ void cu_GEMM_fp32_kernel_64_64_T(const float* __restrict A, const float* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    template<uint32_t L>
    __global__ void cu_GEMM_fp32_F_kernel_64_64_T(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst,
        const float alpha, const float beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);
}
    // FP16
namespace GPUK{
    /**
    * thread = [256, 1]; proc_dims = [128, 128], thread_loc = [4, 16]
    * reg_per_thread = 128 (32-bit); shared_per_block = 16640 Byte
    */
    __global__ void cu_GEMM_fp16_kernel_32_128_128(const __half* __restrict A, const __half* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp16_F_kernel_32_128_128(const __half* __restrict A, const __half* __restrict B, const half* __restrict C, float* __restrict dst,
        const half alpha, const half beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    // reg_per_thread = 126 (32-bit); shared_per_block = 33792 Bytes 
    __global__ void cu_GEMM_fp16_kernel_64_128_128(const __half* __restrict A, const __half* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp16_F_kernel_64_128_128(const __half* __restrict A, const __half* __restrict B, const half* __restrict C, float* __restrict dst,
        const half alpha, const half beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    /**
    * thread = [32, 8]; proc_dims = [128, 128], thread_loc = [4, 16]
    * reg_per_thread = 125 (32-bit); shared_per_block = 32768 Byte
    */
    __global__ void cu_GEMM_fp16_kernel_64_128_128_T(const __half* __restrict A, const __half* __restrict B, float* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp16_F_kernel_64_128_128_T(const __half* __restrict A, const __half* __restrict B, const half* __restrict C, float* __restrict dst,
        const half alpha, const half beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);
}
    // 64b
namespace GPUK{
    // thread = dim3(32, 8)
    // reg_per_thread = 64 (32-bit); shared_per_block = 16896 Byte
    __global__ void cu_GEMM_fp64_kernel_16_64_64(const double* __restrict A, const double* __restrict B, double* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_fp64_F_kernel_16_64_64(const double* __restrict A, const double* __restrict B, const double* __restrict C, double* __restrict dst,
        const double alpha, const double beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);


    // thread = dim3(32, 8)
    // reg_per_thread = 64 (32-bit); shared_per_block = 16896 Byte
    __global__ void cu_GEMM_cplxf_kernel_16_64_64(const double* __restrict A, const double* __restrict B, double* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_cplxf_F_kernel_16_64_64(const double* __restrict A, const double* __restrict B, const double* __restrict C, double* __restrict dst,
        const de::CPf alpha, const de::CPf beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

}
    // CPLXD
namespace GPUK{

    // thread = dim3(32, 8), block_proc_dims = [32, 64]
    // reg_per_thread = 78 (32-bit), shared_per_block = 33792 Bytes
    __global__ void cu_GEMM_cplxd_kernel_16_32_64(const double2* __restrict A, const double2* __restrict B, double2* __restrict dst,
        const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

    __global__ void cu_GEMM_cplxd_F_kernel_16_32_64(const double2* __restrict A, const double2* __restrict B, const double2* __restrict C, double2* __restrict dst,
        const de::CPd alpha, const de::CPd beta, const uint2 proc_dims_v1, const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1);

}
}
}


#endif

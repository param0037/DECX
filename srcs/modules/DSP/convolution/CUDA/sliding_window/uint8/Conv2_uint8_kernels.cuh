/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_UINT8_KERNELS_CUH_
#define _CONV2_UINT8_KERNELS_CUH_


#include "../../../../../core/basic.h"
#include "../../../../../core/utils/decx_cuda_vectypes_ops.cuh"


namespace decx
{
    namespace conv {

        struct _conv2_LKH
        {
            uint _loop, _left;
        };


        static inline decx::conv::_conv2_LKH make_LKH(const uint kernel_height, const uint _loop_frac)
        {
            decx::conv::_conv2_LKH res;
            res._loop = kernel_height / _loop_frac;
            res._left = kernel_height % _loop_frac;
            return res;
        }

        namespace GPUK 
        {
            __device__ __inline__
            decx::utils::_cuda_vec64 reg64_shiftL_uint8(const decx::utils::_cuda_vec64  __x, 
                                                        const decx::utils::_cuda_vec64  __y, 
                                                        const uint                      _count)      // in bits, need to be multipied by 8 if processing uint8_t
            {
                decx::utils::_cuda_vec64 _res;
                size_t _tmp;
                const size_t _select = 18446744073709551615 << (64 - _count);

                _res._ull = __x._ull >> _count;
                _tmp = __y._ull << (64 - _count);

                _res._ull = (_res._ull & (~_select)) | (_tmp & _select);

                return _res;
            }


            __device__ __inline__
            decx::utils::_cuda_vec64 cvt8_fp32_uint8(const float4 __a, const float4 __b)
            {
                decx::utils::_cuda_vec64 _res;

                _res._v_uint8[0] = __float2uint_rn(__a.x);
                _res._v_uint8[1] = __float2uint_rn(__a.y);
                _res._v_uint8[2] = __float2uint_rn(__a.z);
                _res._v_uint8[3] = __float2uint_rn(__a.w);

                _res._v_uint8[4] = __float2uint_rn(__b.x);
                _res._v_uint8[5] = __float2uint_rn(__b.y);
                _res._v_uint8[6] = __float2uint_rn(__b.z);
                _res._v_uint8[7] = __float2uint_rn(__b.w);

                return _res;
            }


            __global__ void
            cu_Conv2_r64x8_uc8_kfp32(const float4* src, const float* kernel, float2* dst, const uint pitch_src, const uint pitch_dst,
                const uint total_ker_len, const uint Wker, const uint2 kernel_shift, const uint2 dst_dims);



            __global__ void
            cu_Conv2_r64x8_uc8_fp32_kfp32(const float4* src, const float* kernel, float4* dst, const uint pitch_src, const uint pitch_dst,
                const uint total_ker_len, const uint Wker, const uint2 kernel_shift, const uint2 dst_dims);



            __global__ void
            cu_Conv2_r64x8_uc8_kfp32_LKH(const float4* src, const float* kernel, float2* dst, const uint pitch_src, const uint pitch_dst,
                const uint Wker, const uint kernel_shift_W, const decx::conv::_conv2_LKH _LKH, const uint2 dst_dims);



            __global__ void
            cu_Conv2_r64x8_uc8_fp32_kfp32_LKH(const float4* src, const float* kernel, float4* dst, const uint pitch_src, const uint pitch_dst,
                const uint Wker, const uint kernel_shift_W, const decx::conv::_conv2_LKH _LKH, const uint2 dst_dims);
        }
    }
}


#endif
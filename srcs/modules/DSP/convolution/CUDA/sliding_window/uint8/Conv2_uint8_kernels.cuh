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

#define _MAX_SLIDE_CONV_R64_V8_U8 8

#define _MAX_SLIDE_CONV_R16_V8_U8_ 4
#define _MAX_SLIDE_CONV_R8_V8_U8_ 2

#define _get_k_loop_w_v8(_max_slide, kernel_shift_W) (_max_slide - 2 * decx::utils::ceil<uint32_t>(kernel_shift_W, 8))


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
                                                const uint8_t                   _count)      // in bits, need to be multipied by 8 if processing uint8_t
    {
        decx::utils::_cuda_vec64 _res;
        uint64_t _tmp;
        const uint64_t _select = 18446744073709551615 << (64 - _count);

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
        const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);



    __global__ void
    cu_Conv2_r64x8_uc8_fp32_kfp32(const float4* src, const float* kernel, float4* dst, const uint pitch_src, const uint pitch_dst,
        const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);



    __global__ void
    cu_Conv2_r64x8_uc8_kfp32_LKH(const float4* src, const float* kernel, float2* dst, const uint pitch_src, const uint pitch_dst,
        const uint Wker, const uint kernel_shift_W, const decx::conv::_conv2_LKH _LKH, const uint2 dst_dims);



    __global__ void
    cu_Conv2_r64x8_uc8_fp32_kfp32_LKH(const float4* src, const float* kernel, float4* dst, const uint pitch_src, const uint pitch_dst,
        const uint Wker, const uint kernel_shift_W, const decx::conv::_conv2_LKH _LKH, const uint2 dst_dims);
}
}
}



namespace decx
{
namespace conv
{
namespace GPUK
{
    __device__ __inline__ static void
    _conv_v8_u8_accu(decx::utils::_cuda_vec64& plane, const float tmp_ker, float4 _accu[2])
    {
        _accu[0].x = __fmaf_rn(__uint2float_rn(plane._v_uint8[0]), tmp_ker, _accu[0].x);
        _accu[0].y = __fmaf_rn(__uint2float_rn(plane._v_uint8[1]), tmp_ker, _accu[0].y);
        _accu[0].z = __fmaf_rn(__uint2float_rn(plane._v_uint8[2]), tmp_ker, _accu[0].z);
        _accu[0].w = __fmaf_rn(__uint2float_rn(plane._v_uint8[3]), tmp_ker, _accu[0].w);

        _accu[1].x = __fmaf_rn(__uint2float_rn(plane._v_uint8[4]), tmp_ker, _accu[1].x);
        _accu[1].y = __fmaf_rn(__uint2float_rn(plane._v_uint8[5]), tmp_ker, _accu[1].y);
        _accu[1].z = __fmaf_rn(__uint2float_rn(plane._v_uint8[6]), tmp_ker, _accu[1].z);
        _accu[1].w = __fmaf_rn(__uint2float_rn(plane._v_uint8[7]), tmp_ker, _accu[1].w);
    }


    __device__ __inline__ static uint8_t
    _pre_residual_v8_conv_u8(decx::utils::_cuda_vec64&   reg_0, 
                               decx::utils::_cuda_vec64&   reg_1, 
                               float4                       _accu[2],
                               const float*                kernel, 
                               const uint8_t                _offset)
    {
        float tmp_ker;

        if (_offset == 0) {
            tmp_ker = kernel[0];
            _conv_v8_u8_accu(reg_0, tmp_ker, _accu);
            return 1;
        }
        decx::utils::_cuda_vec64 plane;
        for (uint8_t k = 0; k < 8; ++k) {
            tmp_ker = (k < _offset - 1) ? 0 : kernel[k - _offset + 1];

            plane = decx::conv::GPUK::reg64_shiftL_uint8(reg_0, reg_1, 8 * (k + 1));
            _conv_v8_u8_accu(plane, tmp_ker, _accu);
        }
        return ((8 - _offset) + 1);
    }



    __device__ __inline__ static void
    _post_residual_v8_conv_u8(decx::utils::_cuda_vec64&  reg_0, 
                                decx::utils::_cuda_vec64&  reg_1, 
                                float4  _accu[2],
                                const float*               kernel, 
                                const uint8_t               _left)
    {
        float tmp_ker;
        
        decx::utils::_cuda_vec64 plane;
        for (uint8_t i = 1; i < 9; ++i) {
            tmp_ker = (i - 1 < _left) ? kernel[i - 1] : 0;

            plane = decx::conv::GPUK::reg64_shiftL_uint8(reg_0, reg_1, 8 * i);
            _conv_v8_u8_accu(plane, tmp_ker, _accu);
        }
    }


    __device__ __inline__ static void
    _full_v8_conv_u8(decx::utils::_cuda_vec64& reg_0, decx::utils::_cuda_vec64& reg_1, float4 _accu[2], 
        const float* kernel)
    {
        float tmp_ker;
        decx::utils::_cuda_vec64 plane;

        for (uint8_t i = 1; i < 9; ++i) {
            tmp_ker = kernel[i - 1];

            plane = decx::conv::GPUK::reg64_shiftL_uint8(reg_0, reg_1, 8 * i);
            _conv_v8_u8_accu(plane, tmp_ker, _accu);
        }
    }
}
}
}



#endif
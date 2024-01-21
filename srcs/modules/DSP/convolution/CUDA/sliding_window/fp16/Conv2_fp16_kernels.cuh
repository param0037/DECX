/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_FP16_KERNELS_CUH_
#define _CONV2_FP16_KERNELS_CUH_


#include "../../../../../core/basic.h"
#include "../../../../../classes/classes_util.h"
#include "../Conv2_kernel_defs.cuh"
#include "../../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../../core/utils/decx_cuda_math_functions.cuh"


#define _MAX_SLIDE_CONV_R16_V8_FP16_ 4
#define _MAX_SLIDE_CONV_R8_V8_FP16_ 2

#define _get_k_loop_w_v8(_max_slide, kernel_shift_W) (_max_slide - 2 * decx::utils::ceil<uint32_t>(kernel_shift_W, 8))



namespace decx
{
namespace conv
{
namespace GPUK
{
    __device__ __inline__
    static void reg_shift_fp16(decx::utils::_cuda_vec128& tmp_reg_ptr)
    {
#if __ABOVE_SM_53
#pragma unroll 7
        for (int i = 1; i < 4; ++i) {
            //__half tmp = ((__half*)tmp_reg_ptr)[i];
            //((__half*)tmp_reg_ptr)[i - 1] = tmp;
            uint32_t _tmp = __funnelshift_r(tmp_reg_ptr._arrui[i - 1], tmp_reg_ptr._arrui[i], 16);
            tmp_reg_ptr._arrui[i - 1] = _tmp;
        }
        tmp_reg_ptr._arrui[3] >>= 16;
#endif
    }


    __device__ __inline__ static void
    _conv_v8_fp16_accu(decx::utils::_cuda_vec128& reg_0, const float tmp_ker, decx::utils::_cuda_vec128 _accu[2])
    {
#if __ABOVE_SM_53
        _accu[0]._vf.x = fmaf(__half2float(reg_0._arrh[0]), tmp_ker, _accu[0]._vf.x);
        _accu[0]._vf.y = fmaf(__half2float(reg_0._arrh[1]), tmp_ker, _accu[0]._vf.y);
        _accu[0]._vf.z = fmaf(__half2float(reg_0._arrh[2]), tmp_ker, _accu[0]._vf.z);
        _accu[0]._vf.w = fmaf(__half2float(reg_0._arrh[3]), tmp_ker, _accu[0]._vf.w);
        _accu[1]._vf.x = fmaf(__half2float(reg_0._arrh[4]), tmp_ker, _accu[1]._vf.x);
        _accu[1]._vf.y = fmaf(__half2float(reg_0._arrh[5]), tmp_ker, _accu[1]._vf.y);
        _accu[1]._vf.z = fmaf(__half2float(reg_0._arrh[6]), tmp_ker, _accu[1]._vf.z);
        _accu[1]._vf.w = fmaf(__half2float(reg_0._arrh[7]), tmp_ker, _accu[1]._vf.w);
#endif
    }


    __device__ __inline__ static uint8_t
    _pre_residual_v8_conv_fp16(decx::utils::_cuda_vec128&   reg_0, 
                               decx::utils::_cuda_vec128&   reg_1, 
                               decx::utils::_cuda_vec128&   _accu,
                               const ushort*                kernel, 
                               const uint8_t                _offset)
    {
#if __ABOVE_SM_53
        uint32_t tmp_ker;

        if (_offset == 0) {
            *((ushort*)&tmp_ker) = kernel[0];
            tmp_ker = __byte_perm(tmp_ker, 0, 0x1010);
            _accu._vf = decx::utils::cuda::__fmah_v8_v1_v8(reg_0._vf, *((__half2*)&tmp_ker), _accu._vf);

            return 1;
        }
        for (uint8_t k = 0; k < 8; ++k) {
            *((ushort*)&tmp_ker) = (k < _offset - 1) ? 0 : kernel[k - _offset + 1];
            tmp_ker = __byte_perm(tmp_ker, 0, 0x1010);

            decx::conv::GPUK::reg_shift_fp16(reg_0);
            reg_0._arrh[7] = reg_1._arrh[k];
            _accu._vf = decx::utils::cuda::__fmah_v8_v1_v8(reg_0._vf, *((__half2*)&tmp_ker), _accu._vf);
        }
        return ((8 - _offset) + 1);
#endif
    }


    __device__ __inline__ static uint8_t
    _pre_residual_v8_conv_fp16_accu(decx::utils::_cuda_vec128&  reg_0, 
                                    decx::utils::_cuda_vec128&  reg_1, 
                                    decx::utils::_cuda_vec128   _accu[2],
                                    const __half*               kernel, 
                                    const uint8_t               _offset)
    {
#if __ABOVE_SM_53
        float tmp_ker;

        if (_offset == 0) {
            tmp_ker = __half2float(kernel[0]);

            decx::conv::GPUK::_conv_v8_fp16_accu(reg_0, tmp_ker, _accu);

            return 1;
        }
        for (uint8_t k = 0; k < 8; ++k) {
            tmp_ker = (k < _offset - 1) ? 0.f : __half2float(kernel[k - _offset + 1]);

            decx::conv::GPUK::reg_shift_fp16(reg_0);
            reg_0._arrh[7] = reg_1._arrh[k];
            decx::conv::GPUK::_conv_v8_fp16_accu(reg_0, tmp_ker, _accu);
        }
        return ((8 - _offset) + 1);
#endif
    }



    __device__ __inline__ static void
    _post_residual_v8_conv_fp16(decx::utils::_cuda_vec128&  reg_0, 
                                decx::utils::_cuda_vec128&  reg_1, 
                                decx::utils::_cuda_vec128&  _accu,
                                const ushort*               kernel, 
                                const uint8_t               _left)
    {
#if __ABOVE_SM_53
        uint32_t tmp_ker;
        
        for (uint8_t i = 1; i < 9; ++i) {
            *((ushort*)&tmp_ker) = (i - 1 < _left) ? kernel[i - 1] : 0;
            tmp_ker = __byte_perm(tmp_ker, 0, 0x1010);

            decx::conv::GPUK::reg_shift_fp16(reg_0);
            reg_0._arrh[7] = reg_1._arrh[i - 1];
            _accu._vf = decx::utils::cuda::__fmah_v8_v1_v8(reg_0._vf, *((__half2*)&tmp_ker), _accu._vf);
        }
#endif
    }


    __device__ __inline__ static void
    _post_residual_v8_conv_fp16_accu(decx::utils::_cuda_vec128& reg_0, decx::utils::_cuda_vec128& reg_1, decx::utils::_cuda_vec128 _accu[2],
            const __half* kernel, const uint8_t _left)
    {
#if __ABOVE_SM_53
        float tmp_ker;
        
        for (uint8_t i = 1; i < 9; ++i) {
            tmp_ker = (i - 1 < _left) ? __half2float(kernel[i - 1]) : 0;

            decx::conv::GPUK::reg_shift_fp16(reg_0);
            reg_0._arrh[7] = reg_1._arrh[i - 1];
            decx::conv::GPUK::_conv_v8_fp16_accu(reg_0, tmp_ker, _accu);
        }
#endif
    }


    __device__ __inline__ static void
    _full_v8_conv_fp16(decx::utils::_cuda_vec128& reg_0, decx::utils::_cuda_vec128& reg_1, decx::utils::_cuda_vec128& _accu, 
        const ushort* kernel)
    {
#if __ABOVE_SM_53
        uint32_t tmp_ker;

        for (uint8_t i = 1; i < 9; ++i) {
            *((ushort*)&tmp_ker) = kernel[i - 1];
            tmp_ker = __byte_perm(tmp_ker, 0, 0x1010);

            decx::conv::GPUK::reg_shift_fp16(reg_0);
            reg_0._arrh[7] = reg_1._arrh[i - 1];
            _accu._vf = decx::utils::cuda::__fmah_v8_v1_v8(reg_0._vf, *((__half2*)&tmp_ker), _accu._vf);
        }
#endif
    }


    __device__ __inline__ static void
    _full_v8_conv_fp16_accu(decx::utils::_cuda_vec128& reg_0, decx::utils::_cuda_vec128& reg_1, decx::utils::_cuda_vec128 _accu[2], 
        const __half* kernel)
    {
#if __ABOVE_SM_53
        float tmp_ker;

        for (uint8_t i = 1; i < 9; ++i) {
            tmp_ker = __half2float(kernel[i - 1]);

            decx::conv::GPUK::reg_shift_fp16(reg_0);
            reg_0._arrh[7] = reg_1._arrh[i - 1];
            decx::conv::GPUK::_conv_v8_fp16_accu(reg_0, tmp_ker, _accu);
        }
#endif
    }
}
}
}


/**
* First load all the necessary values to the shared memory.
*
*         \144halfs(72 floats)     8 halfs
* ----------------------------------
* |                                |        8 halfs
* |        -----------------       |
* |       |                 |      |
* |  apron|     constant    |      |
* |       |                 |      |
* |        -----------------       |
* |                                |
* ----------------------------------
*/


namespace decx {
namespace conv {
namespace GPUK 
{
    __global__
    void cu_hConv2_r8_within(const float4* src, const __half* kernel, float4* dst, 
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);


    __global__
    void cu_hConv2_r168_within(const float4* src, const __half* kernel, float4* dst,
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);


    __global__
    void cu_hConv2_r816_within(const float4* src, const __half* kernel, float4* dst,
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);


    __global__
    void cu_hConv2_r16_within(const float4* src, const __half* kernel, float4* dst,
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);
}

namespace GPUK {
    __global__
    void cu_hConv2_r8_within_accu(const float4* src, const __half* kernel, float4* dst, 
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);


    __global__
    void cu_hConv2_r168_within_accu(const float4* src, const __half* kernel, float4* dst,
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);


    __global__
    void cu_hConv2_r816_within_accu(const float4* src, const __half* kernel, float4* dst,
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);


    __global__
    void cu_hConv2_r16_within_accu(const float4* src, const __half* kernel, float4* dst,
        const uint32_t pitch_src, const uint32_t pitch_dst, const uint2 kernel_dims, const uint2 kernel_shift, const uint2 dst_dims);
}
}
}

#endif
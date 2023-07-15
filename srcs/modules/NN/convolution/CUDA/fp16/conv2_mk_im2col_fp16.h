/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_MK_IM2ROW_FP16_H_
#define _CONV2_MK_IM2ROW_FP16_H_

#include "../../../../classes/Tensor.h"
#include "../../../../classes/TensorArray.h"
#include "../../../../classes/GPU_Tensor.h"
#include "../../../../classes/GPU_TensorArray.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../im2col.cuh"
#include "eq_GEMM_fp16.cuh"
#include "../../../../DSP/convolution/conv_utils.h"
#include "../GPU_conv2_utils.cuh"


namespace decx
{
    namespace conv_I2R {
        static void conv2_MK_im2col_frag_fp16(const float4* src_buf, const float4* ker_buf, float4* I2C_buf, float4* dst,
            decx::conv_I2R::_conv2_I2C_params_set* _im2col_params, decx::cuda_stream* S, const int accu_flag);


        static void conv2_MK_im2col_frag_fp16_stride(const float4* src_buf, const float4* ker_buf, float4* I2C_buf, float4* dst,
            decx::conv_I2R::_conv2_I2C_params_set* _im2col_params, decx::cuda_stream* S, const int accu_flag);

        template <bool _print>
        void conv2_BC_im2col_fp16(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
            decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);

        template <bool _print>
        void conv2_NB_im2col_fp16(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
            decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);

        template <bool _print>
        void conv2_NB_im2col_fp16_stride(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
            decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, const uint2 strideXY, de::DH* handle);

        template <bool _print>
        void conv2_BC_im2col_fp16_stride(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
            decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, const uint2 strideXY, de::DH* handle);
    }
}



static void 
decx::conv_I2R::conv2_MK_im2col_frag_fp16(const float4*                         src_buf,
                                      const float4*                             ker_buf, 
                                      float4*                                   I2C_buf, 
                                      float4*                                   dst,
                                      decx::conv_I2R::_conv2_I2C_params_set*    _im2col_params,
                                      decx::cuda_stream*                        S, 
                                      const int                                 accu_flag)
{
    const uint proc_H = _im2col_params->src_proc_H;

    const size_t I2C_dimH = _im2col_params->HI2C_buf;

    int2 thread_bounds = make_int2(_im2col_params->Wdst_o / 8, proc_H);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.x, 16), decx::utils::ceil<int>(thread_bounds.y, 16));

    decx::conv_I2R::GPUK::cu_Im2Col_v128_fp16 << <grid_I2C, block_I2C, 0, S->get_raw_stream_ref() >> > (src_buf,
                                                                 I2C_buf,
                                                                 thread_bounds,
                                                                 _im2col_params->Wsrc_buf / 8,
                                                                 _im2col_params->WI2C_buf / 8,
                                                                 _im2col_params->ker_dims,
                                                                 _im2col_params->depth / 8);
    
    const ulong2 MatA_load_bounds = make_ulong2(_im2col_params->WI2C_buf / 8, I2C_dimH / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(decx::utils::ceil<size_t>(_im2col_params->WI2C_buf, 256),
        decx::utils::ceil<uint>(_im2col_params->k_tensor_num, 32));

    switch (accu_flag)
    {
    case decx::conv_property::half_conv_ordinary:
        decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16 << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf, 
                                                        ker_buf, 
                                                        dst,
                                                        MatA_load_bounds,
                                                        _im2col_params->Wdst_eqMM / 8,
                                                        _im2col_params->ker_buf_dim.x / 8,
                                                        _im2col_params->ker_buf_dim.x / 128);
        break;
    case decx::conv_property::half_conv_accurate:
        decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16_accu << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf,
                                                        ker_buf, 
                                                        dst,
                                                        MatA_load_bounds,
                                                        _im2col_params->Wdst_eqMM / 8,
                                                        _im2col_params->ker_buf_dim.x / 8,
                                                        _im2col_params->ker_buf_dim.x / 128);
        break;
    default:
        break;
    }
}




static void 
decx::conv_I2R::conv2_MK_im2col_frag_fp16_stride(const float4*                          src_buf,
                                             const float4*                              ker_buf, 
                                             float4*                                    I2C_buf, 
                                             float4*                                    dst,
                                             decx::conv_I2R::_conv2_I2C_params_set*     _im2col_params,
                                             decx::cuda_stream*                         S, 
                                             const int                                  accu_flag)
{
    const uint proc_H = _im2col_params->src_proc_H;

    const size_t I2C_dimH = _im2col_params->HI2C_buf;

    int2 thread_bounds = make_int2(_im2col_params->Wdst_o / 8, proc_H);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.x, 16), decx::utils::ceil<int>(thread_bounds.y, 16));

    decx::conv_I2R::GPUK::cu_Im2Col_v128_stride_fp16 << <grid_I2C, block_I2C, 0, S->get_raw_stream_ref() >> > (src_buf,
                                                                 I2C_buf,
                                                                 _im2col_params->strideXY,
                                                                 thread_bounds,
                                                                 _im2col_params->Wsrc_buf / 8,
                                                                 _im2col_params->WI2C_buf / 8,
                                                                 _im2col_params->ker_dims,
                                                                 _im2col_params->depth / 8);

    const ulong2 MatA_load_bounds = make_ulong2(_im2col_params->WI2C_buf / 8, I2C_dimH / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(decx::utils::ceil<size_t>(_im2col_params->WI2C_buf, 256),
                                             decx::utils::ceil<uint>(_im2col_params->k_tensor_num, 32));

    switch (accu_flag)
    {
    case decx::conv_property::half_conv_ordinary:
        decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16 << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf, 
                                                        ker_buf, 
                                                        dst,
                                                        MatA_load_bounds,
                                                        _im2col_params->Wdst_eqMM / 8,
                                                        _im2col_params->ker_buf_dim.x / 8,
                                                        _im2col_params->ker_buf_dim.x / 128);
        break;
    case decx::conv_property::half_conv_accurate:
        decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16_accu << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf,
                                                        ker_buf, 
                                                        dst,
                                                        MatA_load_bounds,
                                                        _im2col_params->Wdst_eqMM / 8,
                                                        _im2col_params->ker_buf_dim.x / 8,
                                                        _im2col_params->ker_buf_dim.x / 128);
        break;
    default:
        break;
    }
}


#define _I2C_size_fp16_ 1024 * 1024 * 64


#endif
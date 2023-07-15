/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_IM2COL_FP32_H_
#define _CONV2_IM2COL_FP32_H_


#include "../im2row.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/Tensor.h"
#include "../../../../classes/TensorArray.h"
#include "../../../../BLAS/GEMM/CPU/fp32/GEMM_Matrix_B_arrange_fp32.h"
#include "im2row_eq_GEMM_fp32.h"


#define _I2C_frag_size_CPU_ 2048*2048


namespace decx
{
    namespace conv_I2R {
        void _im2row_caller_fp32(const float* src, float* row_buf, float* I2C_buf, const uint _depth_v128,
            const uint2 ker_dims, const uint ker_wp, const uint src_dp_x_wp, const size_t WI2C, const uint2 proc_dims_dst,
            decx::utils::_thread_arrange_1D* t1D);


        void _im2row_caller_fp32_stride(const float* src, float* row_buf, float* I2C_buf, const uint2 strideXY, const uint _depth_v128,
            const uint2 ker_dims, const uint ker_wp, const uint src_dp_x_wp, const size_t WI2C, const uint2 proc_dims_dst,
            decx::utils::_thread_arrange_1D* t1D);


        void conv2_im2row_fp32_NB(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, de::DH* handle);


        void conv2_im2row_fp32_NB_stride(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, const uint2 strideXY, de::DH* handle);



        void conv2_im2row_fp32_BC(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, de::DH* handle);


        void conv2_im2row_fp32_BC_stride(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, const uint2 strideXY, de::DH* handle);
    }
}




#endif
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _IM2COL_FP32_CUH_
#define _IM2COL_FP32_CUH_


#include "../../../../core/basic.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"


#define _IM2COL_D4_FP32_BLOCK_X_ _CUDA_WARP_SIZE_ * 4
#define _IM2COL_D4_FP32_BLOCK_Y_ 2

#define _IM2COL_D8_FP32_BLOCK_X_ _CUDA_WARP_SIZE_ * 4
#define _IM2COL_D8_FP32_BLOCK_Y_ 2

#define _IM2COL_GET_STG_BLOCKDIM_X_(_block_dim_x, _dpitch_v1) ((_block_dim_x / _dpitch_v1) * 4)
#define _IM2COL_GET_STG_BLOCKDIM_Y_(_block_dim_y) (_block_dim_y)


#define _MAX_IM2COL_BUF_SIZE_ 1024 * 1024 * 4


namespace decx
{
namespace nn {
    namespace GPUK 
    {
        struct _cuda_im2col_params;


        __global__ void cu_im2col_NB_fp32(const float4* src, float4* dst, const uint2 dst_dims, const uint2 kernel_dims,
            const uint32_t dpitch_src, const uint32_t wpitch_src, const uint64_t dst_size);


        __global__ void cu_im2col_D4_NB_fp32_divKH(const float4* src, float4* dst, const uint2 dst_dims, const uint2 kernel_dims,
            const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);
    }
}
}


struct decx::nn::GPUK::_cuda_im2col_params
{

};


#endif
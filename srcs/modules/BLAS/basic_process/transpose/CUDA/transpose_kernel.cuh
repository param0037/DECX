/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TRANSPOSE_KERNEL_CUH_
#define _TRANSPOSE_KERNEL_CUH_


#include "../../../../core/basic.h"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"


#define TRANSPOSE_MAT4X4(_loc_transp, tmp)              \
{                                                       \
SWAP(_loc_transp[0].y, _loc_transp[1].x, tmp);          \
SWAP(_loc_transp[0].z, _loc_transp[2].x, tmp);          \
SWAP(_loc_transp[0].w, _loc_transp[3].x, tmp);          \
                                                        \
SWAP(_loc_transp[1].z, _loc_transp[2].y, tmp);          \
                                                        \
SWAP(_loc_transp[1].w, _loc_transp[3].y, tmp);          \
SWAP(_loc_transp[2].w, _loc_transp[3].z, tmp);          \
}




#define TRANSPOSE_MAT8x8(_loc_transp, tmp)                                                  \
{/*row 0, col[1, 7]*/                                                                       \
SWAP(*(((__half*)&_loc_transp[0]) + 1), *(((__half*)&_loc_transp[1])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 2), *(((__half*)&_loc_transp[2])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 3), *(((__half*)&_loc_transp[3])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 4), *(((__half*)&_loc_transp[4])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 5), *(((__half*)&_loc_transp[5])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 6), *(((__half*)&_loc_transp[6])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 7), *(((__half*)&_loc_transp[7])), tmp);                \
/*row1, col[2, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[1]) + 2), *(((__half*)&_loc_transp[2]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 3), *(((__half*)&_loc_transp[3]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 4), *(((__half*)&_loc_transp[4]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 5), *(((__half*)&_loc_transp[5]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 6), *(((__half*)&_loc_transp[6]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 7), *(((__half*)&_loc_transp[7]) + 1), tmp);            \
/*row2, col[3, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[2]) + 3), *(((__half*)&_loc_transp[3]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 4), *(((__half*)&_loc_transp[4]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 5), *(((__half*)&_loc_transp[5]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 6), *(((__half*)&_loc_transp[6]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 7), *(((__half*)&_loc_transp[7]) + 2), tmp);            \
/*row3, col[4, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[3]) + 4), *(((__half*)&_loc_transp[4]) + 3), tmp);            \
SWAP(*(((__half*)&_loc_transp[3]) + 5), *(((__half*)&_loc_transp[5]) + 3), tmp);            \
SWAP(*(((__half*)&_loc_transp[3]) + 6), *(((__half*)&_loc_transp[6]) + 3), tmp);            \
SWAP(*(((__half*)&_loc_transp[3]) + 7), *(((__half*)&_loc_transp[7]) + 3), tmp);            \
/*row4, col[5, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[4]) + 5), *(((__half*)&_loc_transp[5]) + 4), tmp);            \
SWAP(*(((__half*)&_loc_transp[4]) + 6), *(((__half*)&_loc_transp[6]) + 4), tmp);            \
SWAP(*(((__half*)&_loc_transp[4]) + 7), *(((__half*)&_loc_transp[7]) + 4), tmp);            \
/*row5, col[6, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[5]) + 6), *(((__half*)&_loc_transp[6]) + 5), tmp);            \
SWAP(*(((__half*)&_loc_transp[5]) + 7), *(((__half*)&_loc_transp[7]) + 5), tmp);            \
/*row6, col[7, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[6]) + 7), *(((__half*)&_loc_transp[7]) + 6), tmp);            \
}



#define _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ 16



namespace decx
{
    namespace bp {
        namespace GPUK 
        {
            /**
            * Especailly for FFT2D, each thread process 4x4 de::CPf data (sizeof(de::CPf) = sizeof(double)), thus increasing the 
            * throughput
            */
            __global__ void cu_transpose_vec4x4d(double2* src, double2* dst, const uint pitchsrc,
                    const uint pitchdst, const uint2 proc_dim_dst);


            /**
            * Especailly for FFT2D, each thread process 4x4 de::CPf data (sizeof(de::CPf) = sizeof(double)), thus increasing the
            * throughput. Besides, this function also divide the de::CPf data during transposing, designed for IFFT2D
            */
            __global__ void cu_transpose_vec4x4d_and_divide(double2* src, double2* dst, const uint pitchsrc, const uint pitchdst,
                const float signal_len, const uint2 proc_dim_dst);


            /**
            * @param Wsrc : In float4, dev_tmp->width / 4
            * @param Wdst : In float4, dev_tmp->height / 4
            * @param proc_dims : the true dimension of source matrix, measured in element
            */
            __global__ void cu_transpose_vec4x4(const float4* src, float4* dst, const uint Wsrc, const uint Wdst, const uint2 proc_dims);


            /**
            * @param width : In float4, dev_tmp->width / 4
            * @param height : In float4, dev_tmp->height / 4
            * @param true_dim_src : the true dimension of source matrix, measured in element
            */
            __global__ void cu_transpose_vec8x8(const float4* src, float4* dst, const uint width, const uint height, const uint2 true_dim_src);


            /**
            * @param width : In float4, dev_tmp->width / 4
            * @param height : In float4, dev_tmp->height / 4
            */
            __global__ void cu_transpose_vec2x2(const double2* src, double2* dst, const uint width, const uint height, const uint2 proc_dims);
        }
    }
}


#endif
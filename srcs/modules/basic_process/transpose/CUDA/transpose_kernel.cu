/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "transpose_kernel.cuh"



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


#define TRANSPOSE_MAT2X2(_loc_transp, tmp)              \
{                                                       \
SWAP(_loc_transp[0].y, _loc_transp[1].x, tmp);          \
}


#define _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ 16



__global__
void decx::bp::GPUK::cu_transpose_vec4x4d(double2* src,
                          double2 *dst, 
                          const uint pitchsrc,        // in double2 (de::CPf x2)
                          const uint pitchdst,        // in double2 (de::CPf x2)
                          const uint2 proc_dim_dst)   // in de::CPf
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex_src = tidy * 4 * pitchsrc + tidx * 2, 
            dex_dst = tidx * 4 * pitchdst + tidy * 2;

    double2 recv[4][2] = { {decx::utils::vec2_set1_fp64(0), decx::utils::vec2_set1_fp64(0)},
                           {decx::utils::vec2_set1_fp64(0), decx::utils::vec2_set1_fp64(0)},
                           {decx::utils::vec2_set1_fp64(0), decx::utils::vec2_set1_fp64(0)},
                           {decx::utils::vec2_set1_fp64(0), decx::utils::vec2_set1_fp64(0)} };
    double tmp;

    if (tidy * 4 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[0][0] = src[dex_src];          recv[0][1] = src[dex_src + 1];
        dex_src += pitchsrc;
    }
    if (tidy * 4 + 1 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[1][0] = src[dex_src];          recv[1][1] = src[dex_src + 1];
        dex_src += pitchsrc;
    }
    if (tidy * 4 + 2 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[2][0] = src[dex_src];          recv[2][1] = src[dex_src + 1];
        dex_src += pitchsrc;
    }
    if (tidy * 4 + 3 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[3][0] = src[dex_src];          recv[3][1] = src[dex_src + 1];
    }

    SWAP(recv[1][0].x, recv[0][0].y, tmp);
    SWAP(recv[2][0].x, recv[0][1].x, tmp);
    SWAP(recv[3][0].x, recv[0][1].y, tmp);
    SWAP(recv[2][0].y, recv[1][1].x, tmp);
    SWAP(recv[3][0].y, recv[1][1].y, tmp);
    SWAP(recv[3][1].x, recv[2][1].y, tmp);

    if (tidx * 4 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[0][0];          dst[dex_dst + 1] = recv[0][1];
        dex_dst += pitchdst;
    }
    if (tidx * 4 + 1 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[1][0];          dst[dex_dst + 1] = recv[1][1];
        dex_dst += pitchdst;
    }
    if (tidx * 4 + 2 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[2][0];          dst[dex_dst + 1] = recv[2][1];
        dex_dst += pitchdst;
    }
    if (tidx * 4 + 3 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[3][0];          dst[dex_dst + 1] = recv[3][1];
    }
}



__global__
void decx::bp::GPUK::cu_transpose_vec4x4d_and_divide(double2* src,
                                     double2 *dst, 
                                     const uint pitchsrc,        // in double2 (de::CPf x2)
                                     const uint pitchdst,        // in double2 (de::CPf x2)
                                     const float signal_len,
                                     const uint2 proc_dim_dst)   // in de::CPf
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex_src = tidy * 4 * pitchsrc + tidx * 2, 
            dex_dst = tidx * 4 * pitchdst + tidy * 2;

    double2 recv[4][2];
    double tmp;

    if (tidy * 4 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[0][0] = src[dex_src];          recv[0][1] = src[dex_src + 1];
        dex_src += pitchsrc;
    }
    if (tidy * 4 + 1 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[1][0] = src[dex_src];          recv[1][1] = src[dex_src + 1];
        dex_src += pitchsrc;
    }
    if (tidy * 4 + 2 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[2][0] = src[dex_src];          recv[2][1] = src[dex_src + 1];
        dex_src += pitchsrc;
    }
    if (tidy * 4 + 3 < proc_dim_dst.x && tidx < pitchsrc / 2) {
        recv[3][0] = src[dex_src];          recv[3][1] = src[dex_src + 1];
    }

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        ((float4*)&recv[i][0])->x = __fdividef(((float4*)&recv[i][0])->x, signal_len);
        ((float4*)&recv[i][0])->y = __fdividef(((float4*)&recv[i][0])->y, signal_len);
        ((float4*)&recv[i][0])->z = __fdividef(((float4*)&recv[i][0])->z, signal_len);
        ((float4*)&recv[i][0])->w = __fdividef(((float4*)&recv[i][0])->w, signal_len);

        ((float4*)&recv[i][1])->x = __fdividef(((float4*)&recv[i][1])->x, signal_len);
        ((float4*)&recv[i][1])->y = __fdividef(((float4*)&recv[i][1])->y, signal_len);
        ((float4*)&recv[i][1])->z = __fdividef(((float4*)&recv[i][1])->z, signal_len);
        ((float4*)&recv[i][1])->w = __fdividef(((float4*)&recv[i][1])->w, signal_len);
    }

    SWAP(recv[1][0].x, recv[0][0].y, tmp);
    SWAP(recv[2][0].x, recv[0][1].x, tmp);
    SWAP(recv[3][0].x, recv[0][1].y, tmp);
    SWAP(recv[2][0].y, recv[1][1].x, tmp);
    SWAP(recv[3][0].y, recv[1][1].y, tmp);
    SWAP(recv[3][1].x, recv[2][1].y, tmp);

    if (tidx * 4 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[0][0];          dst[dex_dst + 1] = recv[0][1];
        dex_dst += pitchdst;
    }
    if (tidx * 4 + 1 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[1][0];          dst[dex_dst + 1] = recv[1][1];
        dex_dst += pitchdst;
    }
    if (tidx * 4 + 2 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[2][0];          dst[dex_dst + 1] = recv[2][1];
        dex_dst += pitchdst;
    }
    if (tidx * 4 + 3 < proc_dim_dst.y && tidy < pitchdst / 2) {
        dst[dex_dst] = recv[3][0];          dst[dex_dst + 1] = recv[3][1];
    }
}



__global__
/*
* @param Wsrc : In float4, dev_tmp->width / 4
* @param Wdst : In float4, dev_tmp->height / 4
* @param proc_dims : the true dimension of source matrix, measured in element
*/
void decx::bp::GPUK::cu_transpose_vec4x4(const float4* src, float4* dst, const uint Wsrc, const uint Wdst, const uint2 proc_dims)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 _loc_transp[4] = { decx::utils::vec4_set1_fp32(0), decx::utils::vec4_set1_fp32(0),
        decx::utils::vec4_set1_fp32(0), decx::utils::vec4_set1_fp32(0) };
    float tmp;
    size_t dex = 4 * (size_t)tidx * (size_t)Wsrc + (size_t)tidy;

    if (4 * tidx < proc_dims.y && (tidy * 4) < proc_dims.x)          _loc_transp[0] = src[dex];
    dex += (size_t)Wsrc;
    if (4 * tidx + 1 < proc_dims.y && (tidy * 4) < proc_dims.x)      _loc_transp[1] = src[dex];
    dex += (size_t)Wsrc;
    if (4 * tidx + 2 < proc_dims.y && (tidy * 4) < proc_dims.x)      _loc_transp[2] = src[dex];
    dex += (size_t)Wsrc;
    if (4 * tidx + 3 < proc_dims.y && (tidy * 4) < proc_dims.x)      _loc_transp[3] = src[dex];
    dex += (size_t)Wsrc;

    TRANSPOSE_MAT4X4(_loc_transp, tmp);

    dex = 4 * (size_t)tidy * (size_t)Wdst + (size_t)tidx;
    if (4 * tidy < proc_dims.x && (tidx * 4) < proc_dims.y)          dst[dex] = _loc_transp[0];
    dex += (size_t)Wdst;
    if (4 * tidy + 1 < proc_dims.x && (tidx * 4) < proc_dims.y)      dst[dex] = _loc_transp[1];
    dex += (size_t)Wdst;
    if (4 * tidy + 2 < proc_dims.x && (tidx * 4) < proc_dims.y)      dst[dex] = _loc_transp[2];
    dex += (size_t)Wdst;
    if (4 * tidy + 3 < proc_dims.x && (tidx * 4) < proc_dims.y)      dst[dex] = _loc_transp[3];
    dex += (size_t)Wdst;
}



__global__
void decx::bp::GPUK::cu_transpose_vec8x8(const float4* src, float4* dst, const uint Wsrc, const uint Wdst, const uint2 proc_dims)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 _loc_transp[8] = { decx::utils::vec4_set1_fp32(0), decx::utils::vec4_set1_fp32(0),
        decx::utils::vec4_set1_fp32(0), decx::utils::vec4_set1_fp32(0) };
    // If I delete the volatile declaration, the entire register buffer will be set to zero by compiler
    volatile __half tmp;
    size_t dex = 8 * (size_t)tidx * (size_t)Wsrc + (size_t)tidy;

    if (8 * tidx < proc_dims.y && tidy * 8 < proc_dims.x)              _loc_transp[0] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 1 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[1] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 2 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[2] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 3 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[3] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 4 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[4] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 5 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[5] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 6 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[6] = src[dex];
    dex += (size_t)Wsrc;
    if (8 * tidx + 7 < proc_dims.y && tidy * 8 < proc_dims.x)          _loc_transp[7] = src[dex];

    // OPS    -------------------------------------------------------------------------- | -------- Before Being Transposed ----------

    SWAP(*(((__half*)&_loc_transp[0]) + 1), *(((__half*)&_loc_transp[1])), tmp);                // row 0, col[1, 7]
    SWAP(*(((__half*)&_loc_transp[0]) + 2), *(((__half*)&_loc_transp[2])), tmp);
    SWAP(*(((__half*)&_loc_transp[0]) + 3), *(((__half*)&_loc_transp[3])), tmp);
    SWAP(*(((__half*)&_loc_transp[0]) + 4), *(((__half*)&_loc_transp[4])), tmp);
    SWAP(*(((__half*)&_loc_transp[0]) + 5), *(((__half*)&_loc_transp[5])), tmp);
    SWAP(*(((__half*)&_loc_transp[0]) + 6), *(((__half*)&_loc_transp[6])), tmp);
    SWAP(*(((__half*)&_loc_transp[0]) + 7), *(((__half*)&_loc_transp[7])), tmp);

    SWAP(*(((__half*)&_loc_transp[1]) + 2), *(((__half*)&_loc_transp[2]) + 1), tmp);            // row1, col[2, 7]
    SWAP(*(((__half*)&_loc_transp[1]) + 3), *(((__half*)&_loc_transp[3]) + 1), tmp);
    SWAP(*(((__half*)&_loc_transp[1]) + 4), *(((__half*)&_loc_transp[4]) + 1), tmp);
    SWAP(*(((__half*)&_loc_transp[1]) + 5), *(((__half*)&_loc_transp[5]) + 1), tmp);
    SWAP(*(((__half*)&_loc_transp[1]) + 6), *(((__half*)&_loc_transp[6]) + 1), tmp);
    SWAP(*(((__half*)&_loc_transp[1]) + 7), *(((__half*)&_loc_transp[7]) + 1), tmp);

    SWAP(*(((__half*)&_loc_transp[2]) + 3), *(((__half*)&_loc_transp[3]) + 2), tmp);            // row2, col[3, 7]
    SWAP(*(((__half*)&_loc_transp[2]) + 4), *(((__half*)&_loc_transp[4]) + 2), tmp);
    SWAP(*(((__half*)&_loc_transp[2]) + 5), *(((__half*)&_loc_transp[5]) + 2), tmp);
    SWAP(*(((__half*)&_loc_transp[2]) + 6), *(((__half*)&_loc_transp[6]) + 2), tmp);
    SWAP(*(((__half*)&_loc_transp[2]) + 7), *(((__half*)&_loc_transp[7]) + 2), tmp);

    SWAP(*(((__half*)&_loc_transp[3]) + 4), *(((__half*)&_loc_transp[4]) + 3), tmp);            // row3, col[4, 7]
    SWAP(*(((__half*)&_loc_transp[3]) + 5), *(((__half*)&_loc_transp[5]) + 3), tmp);
    SWAP(*(((__half*)&_loc_transp[3]) + 6), *(((__half*)&_loc_transp[6]) + 3), tmp);
    SWAP(*(((__half*)&_loc_transp[3]) + 7), *(((__half*)&_loc_transp[7]) + 3), tmp);

    SWAP(*(((__half*)&_loc_transp[4]) + 5), *(((__half*)&_loc_transp[5]) + 4), tmp);            // row4, col[5, 7]
    SWAP(*(((__half*)&_loc_transp[4]) + 6), *(((__half*)&_loc_transp[6]) + 4), tmp);
    SWAP(*(((__half*)&_loc_transp[4]) + 7), *(((__half*)&_loc_transp[7]) + 4), tmp);

    SWAP(*(((__half*)&_loc_transp[5]) + 6), *(((__half*)&_loc_transp[6]) + 5), tmp);            // row5, col[6, 7]
    SWAP(*(((__half*)&_loc_transp[5]) + 7), *(((__half*)&_loc_transp[7]) + 5), tmp);

    SWAP(*(((__half*)&_loc_transp[6]) + 7), *(((__half*)&_loc_transp[7]) + 6), tmp);            // row6, col[7, 7]

    dex = 8 * (size_t)tidy * (size_t)Wdst + (size_t)tidx;
    if (8 * tidy < proc_dims.x && tidx * 8 < proc_dims.y)              dst[dex] = _loc_transp[0];
    dex += (size_t)Wdst;
    if (8 * tidy + 1 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[1];
    dex += (size_t)Wdst;
    if (8 * tidy + 2 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[2];
    dex += (size_t)Wdst;
    if (8 * tidy + 3 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[3];
    dex += (size_t)Wdst;
    if (8 * tidy + 4 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[4];
    dex += (size_t)Wdst;
    if (8 * tidy + 5 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[5];
    dex += (size_t)Wdst;
    if (8 * tidy + 6 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[6];
    dex += (size_t)Wdst;
    if (8 * tidy + 7 < proc_dims.x && tidx * 8 < proc_dims.y)          dst[dex] = _loc_transp[7];
}




__global__
void decx::bp::GPUK::cu_transpose_vec2x2(const double2* src, double2* dst, const uint width, const uint height, const uint2 proc_dims)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    double2 _loc_transp[2] = { decx::utils::vec2_set1_fp64(0), decx::utils::vec2_set1_fp64(0) };
    double tmp;
    size_t dex = 2 * (size_t)tidx * (size_t)width + (size_t)tidy;

    if (2 * tidx < proc_dims.y && (tidy * 2) < proc_dims.x)              _loc_transp[0] = src[dex];
    dex += (size_t)width;
    if (2 * tidx + 1 < proc_dims.y && (tidy * 2) < proc_dims.x)          _loc_transp[1] = src[dex];

    TRANSPOSE_MAT2X2(_loc_transp, tmp);

    dex = 2 * (size_t)tidy * (size_t)height + (size_t)tidx;
    if (2 * tidy < proc_dims.x && (tidx * 2) < proc_dims.y)              dst[dex] = _loc_transp[0];
    dex += (size_t)height;
    if (2 * tidy + 1 < proc_dims.x && (tidx * 2) < proc_dims.y)          dst[dex] = _loc_transp[1];
}


/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _MAX_POOLING_CUH_
#define _MAX_POOLING_CUH_

#include "../../core/basic.h"


__global__
/*
* @param width : The thread limit on direction of width, in float4
* @param height : The thread limit on direction of height, which is src->height / 2
*/
void cu_max_pooling2x2_f(float4* src, float4* dst, const int width, const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex;

    float4 tmp_rect[4], _max;

    if (tidx < height && tidy < width) {
        dex = tidx * width * 4 + tidy * 2;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        dex += width * 2;
        tmp_rect[2] = src[dex];
        tmp_rect[3] = src[dex + 1];

        _max.x = GetLarger(tmp_rect[0].x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[2].x);
        _max.x = GetLarger(_max.x, tmp_rect[2].y);

        _max.y = GetLarger(tmp_rect[0].z, tmp_rect[0].w);
        _max.y = GetLarger(_max.y, tmp_rect[2].z);
        _max.y = GetLarger(_max.y, tmp_rect[2].w);

        _max.z = GetLarger(tmp_rect[1].x, tmp_rect[1].y);
        _max.z = GetLarger(_max.z, tmp_rect[3].x);
        _max.z = GetLarger(_max.z, tmp_rect[3].y);

        _max.w = GetLarger(tmp_rect[1].z, tmp_rect[1].w);
        _max.w = GetLarger(_max.w, tmp_rect[3].z);
        _max.w = GetLarger(_max.w, tmp_rect[3].w);

        dex = tidx * width + tidy;
        dst[dex] = _max;
    }
}


__global__
/*
* @param width : The thread limit on direction of width, in float4
* @param height : The thread limit on direction of height, which is src->height / 2
*/
void cu_max_pooling2x2_f_with_dex(float4* src, float4* dst, int4 *dex_info_map, const int width, const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex;

    float4 tmp_rect[4], _max;
    int4 _dex_info = make_int4(0, 0, 0, 0);

    if (tidx < height && tidy < width) {
        dex = tidx * width * 4 + tidy * 2;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        dex += width * 2;
        tmp_rect[2] = src[dex];
        tmp_rect[3] = src[dex + 1];

        _max.x = GetLarger(tmp_rect[0].x, tmp_rect[0].y);
        _dex_info.x = tmp_rect[0].x > tmp_rect[0].y ? 0 : 1;
        _max.x = GetLarger(_max.x, tmp_rect[2].x);
        _dex_info.x = _max.x > tmp_rect[2].x ? _dex_info.x : 2;
        _max.x = GetLarger(_max.x, tmp_rect[2].y);
        _dex_info.x = _max.x > tmp_rect[2].y ? _dex_info.x : 3;

        _max.y = GetLarger(tmp_rect[0].z, tmp_rect[0].w);
        _dex_info.y = tmp_rect[0].z > tmp_rect[0].w ? 0 : 1;
        _max.y = GetLarger(_max.y, tmp_rect[2].z);
        _dex_info.y = _max.y > tmp_rect[2].z ? _dex_info.y : 2;
        _max.y = GetLarger(_max.y, tmp_rect[2].w);
        _dex_info.y = _max.y > tmp_rect[2].w ? _dex_info.y : 3;

        _max.z = GetLarger(tmp_rect[1].x, tmp_rect[1].y);
        _dex_info.z = tmp_rect[1].x > tmp_rect[1].y ? 0 : 1;
        _max.z = GetLarger(_max.z, tmp_rect[3].x);
        _dex_info.z = _max.z > tmp_rect[3].x ? _dex_info.z : 2;
        _max.z = GetLarger(_max.z, tmp_rect[3].y);
        _dex_info.z = _max.z > tmp_rect[3].y ? _dex_info.z : 3;

        _max.w = GetLarger(tmp_rect[1].z, tmp_rect[1].w);
        _dex_info.w = tmp_rect[1].z > tmp_rect[1].w ? 0 : 1;
        _max.w = GetLarger(_max.w, tmp_rect[3].z);
        _dex_info.w = _max.w > tmp_rect[3].z ? _dex_info.w : 2;
        _max.w = GetLarger(_max.w, tmp_rect[3].w);
        _dex_info.w = _max.w > tmp_rect[3].w ? _dex_info.w : 3;

        dex = tidx * width + tidy;
        dst[dex] = _max;
        dex_info_map[dex] = _dex_info;
    }
}


__global__
/*
* @param width : The thread limit on direction of width, in float4
* @param height : The thread limit on direction of height, which is src->height / 3
*/
void cu_max_pooling3x2_f(float4* src, float4* dst, const int width, const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex;

    float4 tmp_rect[4], _max;

    if (tidx < height && tidy < width) {
        dex = tidx * width * 6 + tidy * 2;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        dex += width * 2;
        tmp_rect[2] = src[dex];
        tmp_rect[3] = src[dex + 1];

        _max.x = GetLarger(tmp_rect[0].x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[2].x);
        _max.x = GetLarger(_max.x, tmp_rect[2].y);

        _max.y = GetLarger(tmp_rect[0].z, tmp_rect[0].w);
        _max.y = GetLarger(_max.y, tmp_rect[2].z);
        _max.y = GetLarger(_max.y, tmp_rect[2].w);

        _max.z = GetLarger(tmp_rect[1].x, tmp_rect[1].y);
        _max.z = GetLarger(_max.z, tmp_rect[3].x);
        _max.z = GetLarger(_max.z, tmp_rect[3].y);

        _max.w = GetLarger(tmp_rect[1].z, tmp_rect[1].w);
        _max.w = GetLarger(_max.w, tmp_rect[3].z);
        _max.w = GetLarger(_max.w, tmp_rect[3].w);

        dex += width * 2;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];

        _max.x = GetLarger(_max.x, tmp_rect[0].x);
        _max.x = GetLarger(_max.x, tmp_rect[0].y);

        _max.y = GetLarger(_max.y, tmp_rect[0].z);
        _max.y = GetLarger(_max.y, tmp_rect[0].w);

        _max.z = GetLarger(_max.z, tmp_rect[1].x);
        _max.z = GetLarger(_max.z, tmp_rect[1].y);

        _max.w = GetLarger(_max.w, tmp_rect[1].z);
        _max.w = GetLarger(_max.w, tmp_rect[1].w);

        dex = tidx * width + tidy;
        dst[dex] = _max;
    }
}


__global__
/*
* @param width : The thread limit on direction of width, in float4
* @param height : The thread limit on direction of height, which is src->height / 3
*/
void cu_max_pooling3x3_f(float4* src, float4* dst, const int width, const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex;

    float4 tmp_rect[3], _max;

    if (tidx < height && tidy < width) {
        dex = tidx * width * 9 + tidy * 3;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        tmp_rect[2] = src[dex + 2];
        _max.x = GetLarger(tmp_rect[0].x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[0].z);
        _max.y = GetLarger(tmp_rect[0].w, tmp_rect[1].x);
        _max.y = GetLarger(_max.y, tmp_rect[1].y);
        _max.z = GetLarger(tmp_rect[1].z, tmp_rect[1].w);
        _max.z = GetLarger(_max.z, tmp_rect[2].x);
        _max.w = GetLarger(tmp_rect[2].y, tmp_rect[2].z);
        _max.w = GetLarger(_max.w, tmp_rect[2].w);

        dex += width * 3;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        tmp_rect[2] = src[dex + 2];
        _max.x = GetLarger(_max.x, tmp_rect[0].x);
        _max.x = GetLarger(_max.x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[0].z);
        _max.y = GetLarger(_max.y, tmp_rect[0].w);
        _max.y = GetLarger(_max.y, tmp_rect[1].x);
        _max.y = GetLarger(_max.y, tmp_rect[1].y);
        _max.z = GetLarger(_max.z, tmp_rect[1].z);
        _max.z = GetLarger(_max.z, tmp_rect[1].w);
        _max.z = GetLarger(_max.z, tmp_rect[2].x);
        _max.w = GetLarger(_max.w, tmp_rect[2].y);
        _max.w = GetLarger(_max.w, tmp_rect[2].z);
        _max.w = GetLarger(_max.w, tmp_rect[2].w);

        dex += width * 3;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        tmp_rect[2] = src[dex + 2];
        _max.x = GetLarger(_max.x, tmp_rect[0].x);
        _max.x = GetLarger(_max.x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[0].z);
        _max.y = GetLarger(_max.y, tmp_rect[0].w);
        _max.y = GetLarger(_max.y, tmp_rect[1].x);
        _max.y = GetLarger(_max.y, tmp_rect[1].y);
        _max.z = GetLarger(_max.z, tmp_rect[1].z);
        _max.z = GetLarger(_max.z, tmp_rect[1].w);
        _max.z = GetLarger(_max.z, tmp_rect[2].x);
        _max.w = GetLarger(_max.w, tmp_rect[2].y);
        _max.w = GetLarger(_max.w, tmp_rect[2].z);
        _max.w = GetLarger(_max.w, tmp_rect[2].w);

        dex = tidx * width + tidy;
        dst[dex] = _max;
    }
}



__global__
/*
* @param width : The thread limit on direction of width, in float4
* @param height : The thread limit on direction of height, which is src->height / 3
*/
void cu_max_pooling2x3_f(float4* src, float4* dst, const int width, const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex;

    float4 tmp_rect[3], _max;

    if (tidx < height && tidy < width) {
        dex = tidx * width * 6 + tidy * 3;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        tmp_rect[2] = src[dex + 2];
        _max.x = GetLarger(tmp_rect[0].x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[0].z);
        _max.y = GetLarger(tmp_rect[0].w, tmp_rect[1].x);
        _max.y = GetLarger(_max.y, tmp_rect[1].y);
        _max.z = GetLarger(tmp_rect[1].z, tmp_rect[1].w);
        _max.z = GetLarger(_max.z, tmp_rect[2].x);
        _max.w = GetLarger(tmp_rect[2].y, tmp_rect[2].z);
        _max.w = GetLarger(_max.w, tmp_rect[2].w);

        dex += width * 3;
        tmp_rect[0] = src[dex];
        tmp_rect[1] = src[dex + 1];
        tmp_rect[2] = src[dex + 2];
        _max.x = GetLarger(_max.x, tmp_rect[0].x);
        _max.x = GetLarger(_max.x, tmp_rect[0].y);
        _max.x = GetLarger(_max.x, tmp_rect[0].z);
        _max.y = GetLarger(_max.y, tmp_rect[0].w);
        _max.y = GetLarger(_max.y, tmp_rect[1].x);
        _max.y = GetLarger(_max.y, tmp_rect[1].y);
        _max.z = GetLarger(_max.z, tmp_rect[1].z);
        _max.z = GetLarger(_max.z, tmp_rect[1].w);
        _max.z = GetLarger(_max.z, tmp_rect[2].x);
        _max.w = GetLarger(_max.w, tmp_rect[2].y);
        _max.w = GetLarger(_max.w, tmp_rect[2].z);
        _max.w = GetLarger(_max.w, tmp_rect[2].w);

        dex = tidx * width + tidy;
        dst[dex] = _max;
    }
}



__global__
/*
* @param width : The thread limit on direction of width, in float4
* @param height : The thread limit on direction of height, which is src->height / 2
*/
void cu_up_sampling2x2_f(float4* src, float4* dst, int4 *_dex_info_map, const int width, const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex;

    float4 tmp_rect[4], _max;
    int4 _dex_info;

    tmp_rect[0] = make_float4(0, 0, 0, 0);
    tmp_rect[1] = make_float4(0, 0, 0, 0);
    tmp_rect[2] = make_float4(0, 0, 0, 0);
    tmp_rect[3] = make_float4(0, 0, 0, 0);

    if (tidx < height && tidy < width) {
        dex = tidx * width + tidy;
        _max = src[dex];
        _dex_info = _dex_info_map[dex];

        tmp_rect[0].x = (_dex_info.x == 0) ? _max.x : 0;
        tmp_rect[0].y = (_dex_info.x == 1) ? _max.x : 0;
        tmp_rect[2].x = (_dex_info.x == 2) ? _max.x : 0;
        tmp_rect[2].y = (_dex_info.x == 3) ? _max.x : 0;

        tmp_rect[0].z = (_dex_info.y == 0) ? _max.y : 0;
        tmp_rect[0].w = (_dex_info.y == 1) ? _max.y : 0;
        tmp_rect[2].z = (_dex_info.y == 2) ? _max.y : 0;
        tmp_rect[2].w = (_dex_info.y == 3) ? _max.y : 0;

        tmp_rect[1].x = (_dex_info.z == 0) ? _max.z : 0;
        tmp_rect[1].y = (_dex_info.z == 1) ? _max.z : 0;
        tmp_rect[3].x = (_dex_info.z == 2) ? _max.z : 0;
        tmp_rect[3].y = (_dex_info.z == 3) ? _max.z : 0;

        tmp_rect[1].z = (_dex_info.w == 0) ? _max.w : 0;
        tmp_rect[1].w = (_dex_info.w == 1) ? _max.w : 0;
        tmp_rect[3].z = (_dex_info.w == 2) ? _max.w : 0;
        tmp_rect[3].w = (_dex_info.w == 3) ? _max.w : 0;

        dex = tidx * width * 4 + tidy * 2;
        dst[dex] = tmp_rect[0];
        dst[dex + 1] = tmp_rect[1];
        dex += width * 2;
        dst[dex] = tmp_rect[2];
        dst[dex + 1] = tmp_rect[3];
    }
}


#endif
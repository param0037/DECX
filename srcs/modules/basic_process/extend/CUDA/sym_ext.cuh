/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _SYM_EXT_CUH_
#define _SYM_EXT_CUH_


#include "../../../core/basic.h"


__global__
/*
* @param src : The data space should be loaded the original matrix
* at the ccenter first. THIS POINTER IS WHERE THE PROCESS AREA BEGINS.
* @param bounds.x : The width of workspace, in float4
* @param bounds.y : The height of workspace, in element
* @param pitch : The pitch of device buffer(src)
*/
void cu_sym_ext_T_vec4(float4* src, uint2 bounds, size_t pitch)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint H_map = 2 * bounds.y - tidx;
    size_t addr = (size_t)H_map * pitch + (size_t)tidy;
    float4 tmp;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmp = src[addr];
        addr = (size_t)tidx * pitch + (size_t)tidy;
        src[addr] = tmp;
    }
}


__global__
/*
* @param src : The data space should be loaded the original matrix
* at the ccenter first. THIS POINTER KEEPS THE SAME AS THAT IN THE KERNEL ABOVE.
* @param bounds.x : The width of workspace, in element
* @param bounds.y : The height of workspace, in element
* @param pitch : The pitch of device buffer(src)
* @param height : The total height from the very top to the begin of processed area
*/
void cu_sym_ext_B_vec4(float4* src, uint2 bounds, size_t pitch, uint height)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint H_map = height - 2 - tidx;
    size_t addr = (size_t)H_map * pitch + (size_t)tidy;
    float4 tmp;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmp = src[addr];
        addr = (size_t)(tidx + height) * pitch + (size_t)tidy;
        src[addr] = tmp;
    }
}


__global__
/*
* @param src : The data space should be loaded the original matrix
* at the ccenter first. THIS POINTER IS WHERE THE PROCESS AREA BEGINS.
* @param bounds.x : The width of workspace, in float4
* @param bounds.y : The height of workspace, in element
* @param pitch : The pitch of device buffer(src)
*/
void cu_sym_ext_L(float* src, uint2 bounds, size_t pitch)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint W_map = 2 * bounds.x - tidy;
    size_t addr = (size_t)tidx * pitch + (size_t)W_map;
    float tmp;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmp = src[addr];
        addr = (size_t)tidx * pitch + (size_t)tidy;
        src[addr] = tmp;
    }
}


__global__
/*
* @param src : The data space should be loaded the original matrix
* at the ccenter first. THIS POINTER KEEPS THE SAME AS THAT IN THE KERNEL ABOVE.
* @param bounds.x : The width of workspace, in element
* @param bounds.y : The height of workspace, in element
* @param pitch : The pitch of device buffer(src)
* @param height : The total height from the very top to the begin of processed area
*/
void cu_sym_ext_R(float* src, uint2 bounds, size_t pitch, uint width)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint W_map = width - 2 - tidy;
    size_t addr = (size_t)tidx * pitch + (size_t)W_map;
    float tmp;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmp = src[addr];
        addr = (size_t)tidx * pitch + (size_t)(tidy + width);
        src[addr] = tmp;
    }
}


#endif
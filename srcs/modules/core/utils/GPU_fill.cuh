/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GPU_FILL_CUH_
#define _GPU_FILL_CUH_


#include "../cudaStream_management/cudaStream_queue.h"
#include "../../classes/classes_util.h"
#include "leftovers.h"
#include "../basic.h"

__global__
void cu_fill1D_const_mem_vec4_fp32(float4 *dst, size_t len)
{
    float4 reg;
    float _fill;
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    _fill = ((float*)&Const_Mem[0])[0];
    reg = make_float4(_fill, _fill, _fill, _fill);
    if (tid < len) {
        dst[tid] = reg;
    }
}


__global__
void cu_fill1D_internal_vec4_fp32(float4* src, size_t len, decx::utils::_left_4 _L_info)
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    float reg;
    int i_reg;
    float4 _fill;
    
    reg = ((float*)(&Const_Mem[0]))[0];

    if (tid == 0) {
        _fill = src[tid];
        for (i_reg = 4 - _L_info._occupied_len; i_reg < 4; ++i_reg) {
            ((float*)&_fill)[i_reg] = reg;
        }
    }
    else {
        _fill = make_float4(reg, reg, reg, reg);
    }

    if (tid < len) {
        src[tid] = _fill;
    }
}


__global__
void cu_fill2D_internal_vec4_fp32(float4* src, size_t pitch, uint2 proc_dim, decx::utils::_left_4 _WL_info)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex = tidx * pitch + tidy;

    float reg;
    int i_reg;
    float4 _fill;

    reg = ((float*)(&Const_Mem[0]))[0];

    if (tidy == 0) {
        _fill = src[dex];
        for (i_reg = 4 - _WL_info._occupied_len; i_reg < 4; ++i_reg) {
            ((float*)&_fill)[i_reg] = reg;
        }
    }
    else {
        _fill = make_float4(reg, reg, reg, reg);
    }

    if (tidx < proc_dim.y && tidy < proc_dim.x) {
        src[dex] = _fill;
    }
}


namespace decx
{
    namespace utils
    {
        /*
        * This function fill the blank with one of the exsiting elements
        * @param src : in float4, the location will be filled
        * @param total_len : the total length of memory space
        * @param len : how much elelment to be filled, in float
        */
        static void 
        gpu_internal_fill1D_fp32(float4* src, size_t total_len, size_t len, decx::cuda_stream* S);


        static void 
        gpu_fill1D_fp32(float4 *dst, size_t len, decx::cuda_stream* S);


        /*
        * This function fill the blank with one of the exsiting elements
        * @param src : in float4, the location will be filled
        * @param total_len : the total length of memory space
        * @param len : how much elelment to be filled, in float
        */
        static void
        gpu_internal_fill2D_fp32(float4* src, size_t pitch, size_t width, uint height, decx::cuda_stream* S);
    }
}


static void
decx::utils::gpu_internal_fill1D_fp32(float4* src, size_t total_len, size_t len, decx::cuda_stream* S)
{
    if (len != 0)
    {
        size_t offset, actual_fill_len;
        if (len % 4) {
            actual_fill_len = decx::utils::ceil<size_t>(len, 4);
            offset = total_len / 4 - actual_fill_len;
        }
        else {
            actual_fill_len = len / 4 + 1;
            offset = total_len / 4 - actual_fill_len;
        }

        const uint max_th = decx::cuP.prop.maxThreadsPerBlock;
        const uint grid = decx::utils::ceil<size_t>(len, max_th);

        decx::utils::_left_4 _L_info;
        decx::utils::_left_4_advisor(&_L_info, len);

        if (S == NULL) {
            cu_fill1D_internal_vec4_fp32 << <grid, max_th >> > (src + offset, actual_fill_len, _L_info);
        }
        else {
            cu_fill1D_internal_vec4_fp32 << <grid, max_th, 0, S->get_raw_stream_ref() >> > (
                src + offset, actual_fill_len, _L_info);
        }
    }
    else {
        return;
    }
}



static void
decx::utils::gpu_fill1D_fp32(float4* dst, size_t len, decx::cuda_stream* S)
{
    if (len != 0)
    {
        const uint max_th = decx::cuP.prop.maxThreadsPerBlock;
        const uint grid = decx::utils::ceil<size_t>(len, max_th);

        if (S == NULL) {
            cu_fill1D_const_mem_vec4_fp32 << <grid, max_th >> > (dst, len);
        }
        else {
            cu_fill1D_const_mem_vec4_fp32 << <grid, max_th, 0, S->get_raw_stream_ref() >> > (
                dst, len);
        }
    }
}



static void
decx::utils::gpu_internal_fill2D_fp32(float4* src, size_t pitch, size_t width, uint height, decx::cuda_stream* S)
{
    size_t w_offset, w_actual_fill_len;
    if (width % 4) {
        w_actual_fill_len = decx::utils::ceil<size_t>(width, 4);
        w_offset = pitch / 4 - w_actual_fill_len;
    }
    else {
        w_actual_fill_len = width / 4 + 1;
        w_offset = pitch / 4 - w_actual_fill_len;
    }
    dim3 thread(32, 8);
    dim3 grid(decx::utils::ceil<uint>(height, 32), decx::utils::ceil<uint>(w_actual_fill_len, 8));

    decx::utils::_left_4 _L_info;
    decx::utils::_left_4_advisor(&_L_info, width);

    if (S == NULL) {
        cu_fill2D_internal_vec4_fp32 << <grid, thread >> > (
            src + w_offset, pitch, make_uint2(w_actual_fill_len, height), _L_info);
    }
    else {
        cu_fill2D_internal_vec4_fp32 << <grid, thread, 0, S->get_raw_stream_ref() >> > (
            src + w_offset, pitch, make_uint2(w_actual_fill_len, height), _L_info);
    }
}


//namespace decx
//{
//    namespace utils
//    {
//        static void gpu_internal_fill1D_D_buffer(float4* dev_A, float4 *dev_B)
//    }
//}



#endif
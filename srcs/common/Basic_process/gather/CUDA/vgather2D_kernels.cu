/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

#include "cuda_gather_kernels.cuh"
#include "../../../CUSV/decx_cuda_vectypes_ops.cuh"
#include "../../../CUSV/decx_cuda_math_functions.cuh"


namespace decx
{
namespace GPUK{
    __global__ void cu_vgather2D_fp32(cudaTextureObject_t tex, const float2* map, float4* dst,
        const uint2 src_dims_v1, const uint2 proc_dims, const uint32_t pitchmap_v1, const uint32_t pitchdst_v);


    __global__ void cu_vgather2D_uint8(cudaTextureObject_t tex, const float2* map, uchar4* dst,
        const uint2 src_dims_v1, const uint2 proc_dims, const uint32_t pitchmap_v1, const uint32_t pitchdst_v);
}
}


__global__ void decx::GPUK::
cu_vgather2D_fp32(cudaTextureObject_t   tex, 
                  const float2*         map, 
                  float4*               dst,
                  const uint2           src_dims_v1, 
                  const uint2           proc_dims_v, 
                  const uint32_t        pitchmap_v1, 
                  const uint32_t        pitchdst_v)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    decx::utils::_cuda_vec128 reg;
    float2 coordinates;

    const uint64_t dex_map = tidy * pitchmap_v1 + tidx * 4;
    const uint64_t dex_dst = tidy * pitchdst_v + tidx;

    if (tidx < proc_dims_v.x && tidy < proc_dims_v.y)
    {
        int32_t predX = 0, predY = 0, pred_all = 0;

        if (tidx * 4 < pitchmap_v1) {
            coordinates = map[dex_map];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            reg._vf.x = tex2D<float>(tex, coordinates.x, coordinates.y);
            reg._vi.x = reg._vi.x & pred_all;
        }

        if (tidx * 4 + 1 < pitchmap_v1) {
            coordinates = map[dex_map + 1];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            reg._vf.y = tex2D<float>(tex, coordinates.x, coordinates.y);
            reg._vi.y = reg._vi.y & pred_all;
        }

        if (tidx * 4 + 2 < pitchmap_v1) {
            coordinates = map[dex_map + 2];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            reg._vf.z = tex2D<float>(tex, coordinates.x, coordinates.y);
            reg._vi.z = reg._vi.z & pred_all;
        }
        
        if (tidx * 4 + 3 < pitchmap_v1) {
            coordinates = map[dex_map + 3];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            reg._vf.w = tex2D<float>(tex, coordinates.x, coordinates.y);
            reg._vi.w = reg._vi.w & pred_all;
        }

        dst[dex_dst] = reg._vf;
    }
}



__global__ void decx::GPUK::
cu_vgather2D_uint8(cudaTextureObject_t   tex, 
                  const float2*         map, 
                  uchar4*               dst,
                  const uint2           src_dims_v1, 
                  const uint2           proc_dims_v, 
                  const uint32_t        pitchmap_v1, 
                  const uint32_t        pitchdst_v)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uchar4 reg;
    float2 coordinates;
    float normalized_res;

    const uint64_t dex_map = tidy * pitchmap_v1 + tidx * 4;
    const uint64_t dex_dst = tidy * pitchdst_v + tidx;

    if (tidx < proc_dims_v.x && tidy < proc_dims_v.y)
    {
        int8_t predX = 0, predY = 0, pred_all = 0;

        if (tidx * 4 < pitchmap_v1){
            coordinates = map[dex_map];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            normalized_res = tex2D<float>(tex, coordinates.x, coordinates.y);
            normalized_res = __fmul_rn(normalized_res, 255.f);
            reg.x = __float2int_rn(normalized_res) & pred_all;
        }

        if (tidx * 4 + 1 < pitchmap_v1){
            coordinates = map[dex_map + 1];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            normalized_res = tex2D<float>(tex, coordinates.x, coordinates.y);
            normalized_res = __fmul_rn(normalized_res, 255.f);
            reg.y = __float2int_rn(normalized_res) & pred_all;
        }

        if (tidx * 4 + 2< pitchmap_v1){
            coordinates = map[dex_map + 2];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            normalized_res = tex2D<float>(tex, coordinates.x, coordinates.y);
            normalized_res = __fmul_rn(normalized_res, 255.f);
            reg.z = __float2int_rn(normalized_res) & pred_all;
        }

        if (tidx * 4 + 3 < pitchmap_v1){
            coordinates = map[dex_map + 3];
            predX = (coordinates.x > src_dims_v1.x - 1 || coordinates.x < 0);
            predX = predX - 1;          // 0xFFFFFFFF if pred is 0, 0 otherwise
            predY = (coordinates.y > src_dims_v1.y - 1 || coordinates.y < 0);
            predY = predY - 1;
            pred_all = predX & predY;
            coordinates.x = __fadd_rn(coordinates.x, 0.5f);
            coordinates.y = __fadd_rn(coordinates.y, 0.5f);
            normalized_res = tex2D<float>(tex, coordinates.x, coordinates.y);
            normalized_res = __fmul_rn(normalized_res, 255.f);
            reg.w = __float2int_rn(normalized_res) & pred_all;
        }

        dst[dex_dst] = reg;
    }
}


void decx::GPUK::vgather2D_fp32(cudaTextureObject_t tex,        const float2* map,              
                                float* dst,                     const uint2 src_dims_v1,        
                                const uint2 proc_dims,          const uint32_t pitchmap_v1,     
                                const uint32_t pitchdst_v,      dim3 block,
                                dim3 grid,                      decx::cuda_stream* S)
{
    decx::GPUK::cu_vgather2D_fp32<<<grid, block, 0, S->get_raw_stream_ref()>>>(
        tex, map, (float4*)dst, src_dims_v1, proc_dims, pitchmap_v1, pitchdst_v);
}


void decx::GPUK::vgather2D_uint8(cudaTextureObject_t tex,        const float2* map,              
                                 uint8_t* dst,                   const uint2 src_dims_v1,        
                                 const uint2 proc_dims,          const uint32_t pitchmap_v1,     
                                 const uint32_t pitchdst_v,      dim3 block,
                                 dim3 grid,                      decx::cuda_stream* S)
{
    decx::GPUK::cu_vgather2D_uint8<<<grid, block, 0, S->get_raw_stream_ref()>>>(
        tex, map, (uchar4*)dst, src_dims_v1, proc_dims, pitchmap_v1, pitchdst_v);
}

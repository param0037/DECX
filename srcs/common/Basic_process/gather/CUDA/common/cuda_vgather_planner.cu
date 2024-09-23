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

#include "cuda_vgather_planner.h"
#include "../cuda_gather_kernels.cuh"
#include "../../VGT_kernels_LUT_selector.h"


namespace decx{
namespace GPUK{
    static void* cu_vgather2D_kernels[3][3] = {
        {   (void*)decx::GPUK::vgather2D_fp32,
            NULL,       // type_out = uint8_t
        }, {
            NULL,       // type_out = float
            (void*)decx::GPUK::vgather2D_uint8,
        }, {
            NULL,
            NULL,       // type_out = float
            (void*)decx::GPUK::vgather2D_uchar4,
        }
    };
}
}


cudaTextureFilterMode decx::get_filter_mode_by_intp_type(const de::Interpolate_Types type)
{
    switch (type)
    {
    case de::Interpolate_Types::INTERPOLATE_BILINEAR:
        return cudaFilterModeLinear;
        break;

    case de::Interpolate_Types::INTERPOLATE_NEAREST:
        return cudaFilterModePoint;
        break;
    
    default:
        return cudaFilterModeLinear;
        break;
    }
}


uint2 decx::cuda_VGT2D_planner::get_src_dims_v1() const
{
    return make_uint2(this->_res_desc.res.pitch2D.width, this->_res_desc.res.pitch2D.height);
}


decx::cuda_VGT2D_planner::cuda_VGT2D_planner()
{
    memset(&this->_tex_desc, 0, sizeof(cudaTextureDesc));
    memset(&this->_res_desc, 0, sizeof(cudaResourceDesc));
    memset(&this->_channel_fmt, 0, sizeof(cudaChannelFormatDesc));

    this->_texture = 0;
}


template <typename _type_in, typename _type_out>
void decx::cuda_VGT2D_planner::plan(const de::Interpolate_Types type, 
    const decx::_matrix_layout* src_layout, const decx::_matrix_layout* dst_layout)
{
    // this->_intp_type = type;

    // this->_proc_dims = make_uint2(dst_layout->width, dst_layout->height);

    // this->_type_in_size = sizeof(_type_in);
    // this->_type_out_size = sizeof(_type_out);
    // this->plan_alignment();

    // this->_proc_w_v = decx::utils::ceil<uint32_t>(this->_proc_dims.x, this->_alignment);

    decx::cuda_ElementWise2D_planner::plan(make_uint2(dst_layout->width, dst_layout->height), sizeof(_type_in), sizeof(_type_out));

    this->_channel_fmt = cudaCreateChannelDesc<_type_in>();

    this->_res_desc.resType = cudaResourceTypePitch2D;
    this->_res_desc.res.pitch2D.desc = _channel_fmt;
    this->_res_desc.res.pitch2D.width = src_layout->width;
    this->_res_desc.res.pitch2D.height = src_layout->height;
    this->_res_desc.res.pitch2D.pitchInBytes = src_layout->pitch * sizeof(_type_in);

    this->_tex_desc.addressMode[0] = cudaAddressModeWrap;
    this->_tex_desc.addressMode[1] = cudaAddressModeWrap;
    this->_tex_desc.filterMode = decx::get_filter_mode_by_intp_type(this->_intp_type);
    if (std::is_same<_type_in, float>::value){
        this->_tex_desc.readMode = cudaReadModeElementType;
    }
    else{
        this->_tex_desc.readMode = cudaReadModeNormalizedFloat;
    }
    this->_tex_desc.normalizedCoords = 0;

    // this->_block = dim3(32, 32);
    // this->_grid = dim3(decx::utils::ceil<uint32_t>(this->_proc_w_v, 32),
    //                    decx::utils::ceil<uint32_t>(this->_proc_dims.y, 8));
}

template void decx::cuda_VGT2D_planner::plan<float, float>(const de::Interpolate_Types, const decx::_matrix_layout*, const decx::_matrix_layout*);
template void decx::cuda_VGT2D_planner::plan<uint8_t, uint8_t>(const de::Interpolate_Types, const decx::_matrix_layout*, const decx::_matrix_layout*);
template void decx::cuda_VGT2D_planner::plan<uchar4, uchar4>(const de::Interpolate_Types, const decx::_matrix_layout*, const decx::_matrix_layout*);


template <typename _type_in, typename _type_out>
void decx::cuda_VGT2D_planner::run(const _type_in* src,             const float2* map, 
                                   _type_out* dst,                  const uint32_t pitchmap_v1, 
                                   const uint32_t pitchdst_v1,      decx::cuda_stream* S)
{
    this->_res_desc.res.pitch2D.devPtr = (void*)src;
    uint2 selector = decx::VGT2D_kernel_selector<_type_in, _type_out>();
    cudaCreateTextureObject(&this->_texture, &this->_res_desc, &this->_tex_desc, NULL);

    auto* p_kernel = (decx::GPUK::cuda_vgather_kernel<_type_out>*)decx::GPUK::cu_vgather2D_kernels[selector.x][selector.y];

    (*p_kernel)(this->_texture, map, (_type_out*)dst, this->get_src_dims_v1(), 
        make_uint2(this->_proc_w_v, this->_proc_dims.y), pitchmap_v1, 
        pitchdst_v1 / this->_alignment, this->_block, this->_grid, S);
}

template void decx::cuda_VGT2D_planner::run<float, float>(const float*, const float2*, float*, const uint32_t, const uint32_t, decx::cuda_stream*);
template void decx::cuda_VGT2D_planner::run<uint8_t, uint8_t>(const uint8_t*, const float2*, uint8_t*, const uint32_t, const uint32_t, decx::cuda_stream*);
template void decx::cuda_VGT2D_planner::run<uchar4, uchar4>(const uchar4*, const float2*, uchar4*, const uint32_t, const uint32_t, decx::cuda_stream*);


void decx::cuda_VGT2D_planner::release(decx::cuda_VGT2D_planner* _fake_this)
{
    return;
}

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

#ifndef _CUDA_VGATHER_PLANNER_H_
#define _CUDA_VGATHER_PLANNER_H_

#include "../../../../basic.h"
#include "../../../../Element_wise/common/cuda_element_wise_planner.h"
#include "../../interpolate_types.h"
#include "../../../../Classes/GPU_Matrix.h"


namespace decx
{
    class cuda_VGT2D_planner;


    cudaTextureFilterMode get_filter_mode_by_intp_type(const de::Interpolate_Types type);
}


class decx::cuda_VGT2D_planner : public decx::cuda_ElementWise2D_planner
{
private:
    cudaTextureObject_t _texture;
    cudaResourceDesc _res_desc;
    cudaTextureDesc _tex_desc;
    cudaChannelFormatDesc _channel_fmt;


    de::Interpolate_Types _intp_type;

public:
    cuda_VGT2D_planner();

    
    template <typename _type_in, typename _type_out>
    void plan(const de::Interpolate_Types type, const decx::_matrix_layout* src_layout, const decx::_matrix_layout* dst_layout);


    template <typename _type_in, typename _type_out>
    void run(const _type_in* src, const float2* map, _type_out* dst, const uint32_t pitchmap_v1, const uint32_t pitchdst_v1,
        decx::cuda_stream* S);


    static void release(decx::cuda_VGT2D_planner* _fake_this);

private:
    uint2 get_src_dims_v1() const;
};


#endif

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

#include "../resample.h"
#include "../../../../common/Basic_process/gather/CUDA/common/cuda_vgather_planner.h"


_DECX_API_ void de::dsp::cuda::Resample(de::InputGPUMatrix src, de::InputGPUMatrix map, de::OutputGPUMatrix dst,
    de::Interpolate_Types interpoate_mode)
{
    de::ResetLastError();

    const decx::_GPU_Matrix* _src = dynamic_cast<const decx::_GPU_Matrix*>(&src);
    const decx::_GPU_Matrix* _map = dynamic_cast<const decx::_GPU_Matrix*>(&map);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    // decx::cuda_VGT2D_planner planner;
    // planner.plan<float, float>(interpoate_mode, &_src->get_layout(), &_dst->get_layout());

    // planner.run<float, float>((float*)_src->Mat.ptr, (float2*)_map->Mat.ptr, (float*)_dst->Mat.ptr, _map->Pitch(), _dst->Pitch(), S);

    decx::cuda_VGT2D_planner planner;
    planner.plan<uint8_t, uint8_t>(interpoate_mode, &_src->get_layout(), &_dst->get_layout());

    planner.run<uint8_t, uint8_t>((uint8_t*)_src->Mat.ptr, (float2*)_map->Mat.ptr, (uint8_t*)_dst->Mat.ptr, _map->Pitch(), _dst->Pitch(), S);

    E->event_record(S);
    E->synchronize();

    E->detach();
    S->detach();
}
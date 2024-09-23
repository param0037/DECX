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
#include "../../../core/resources_manager/decx_resource.h"

namespace decx{
    static decx::ResourceHandle cu_VGT2D;

    template <typename _type_in, typename _type_out>
    static void resample2D_caller(const decx::_GPU_Matrix* src, const decx::_GPU_Matrix* map, decx::_GPU_Matrix* dst, 
            de::Interpolate_Types interpolate_mode, de::DH* handle);
}


template <typename _type_in, typename _type_out>
void decx::resample2D_caller(const decx::_GPU_Matrix* src, const decx::_GPU_Matrix* map, decx::_GPU_Matrix* dst, 
    de::Interpolate_Types interpolate_mode, de::DH* handle)
{
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

    decx::cu_VGT2D.lock();

    auto* planner = decx::cu_VGT2D.get_resource_raw_ptr<decx::cuda_VGT2D_planner>();
    planner->plan<_type_in, _type_out>(interpolate_mode, &src->get_layout(), &dst->get_layout());
    
    planner->run<_type_in, _type_out>((_type_in*)src->Mat.ptr, (float2*)map->Mat.ptr, (_type_out*)dst->Mat.ptr, map->Pitch(), dst->Pitch(), S);

    E->event_record(S);
    E->synchronize();

    E->detach();
    S->detach();

    decx::cu_VGT2D.unlock();
}


_DECX_API_ void de::dsp::cuda::Resample(de::InputGPUMatrix src, de::InputGPUMatrix map, de::OutputGPUMatrix dst,
    de::Interpolate_Types interpoate_mode)
{
    de::ResetLastError();

    const decx::_GPU_Matrix* _src = dynamic_cast<const decx::_GPU_Matrix*>(&src);
    const decx::_GPU_Matrix* _map = dynamic_cast<const decx::_GPU_Matrix*>(&map);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    if (decx::cu_VGT2D._res_ptr == NULL){
        decx::cu_VGT2D.RegisterResource(new decx::cuda_VGT2D_planner, 5, decx::cuda_VGT2D_planner::release);
    }

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::resample2D_caller<float, float>(_src, _map, _dst, interpoate_mode, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::resample2D_caller<uint8_t, uint8_t>(_src, _map, _dst, interpoate_mode, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UCHAR4_:
        decx::resample2D_caller<uchar4, uchar4>(_src, _map, _dst, interpoate_mode, de::GetLastError());
        break;
    
    default:
        break;
    }

}
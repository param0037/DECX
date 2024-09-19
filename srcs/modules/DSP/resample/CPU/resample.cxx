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

#include "../../../../common/Basic_process/gather/CPU/common/cpu_vgather_planner.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../../common/FMGR/fragment_arrangment.h"
#include "../resample.h"
#include "../../../../common/Basic_process/gather/CPU/gather_kernels.h"
#include "../../../core/resources_manager/decx_resource.h"


static decx::ResourceHandle g_VGT2D_hdlr;


namespace decx{
namespace dsp
{
    template <typename _type_in, typename _type_out>
    static void resample_caller(const decx::_Matrix* src, const decx::_Matrix* map, decx::_Matrix* dst, 
        const de::Interpolate_Types intp_type, de::DH* handle);
}
}


template <typename _type_in, typename _type_out> static void decx::dsp::
resample_caller(const decx::_Matrix* src, const decx::_Matrix* map, decx::_Matrix* dst, const de::Interpolate_Types intp_type, de::DH* handle)
{
    if (g_VGT2D_hdlr._res_ptr == NULL){
        g_VGT2D_hdlr.RegisterResource(new decx::cpu_VGT2D_planner, 5, decx::cpu_VGT2D_planner::release);
    }

    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    auto* VGT = g_VGT2D_hdlr.get_resource_raw_ptr<decx::cpu_VGT2D_planner>();

    g_VGT2D_hdlr.lock();

    VGT->plan(decx::cpu::_get_permitted_concurrency(), make_uint2(dst->Width(), dst->Height()), 
        sizeof(uint8_t), intp_type, make_uint2(src->Width(), src->Height()), 
        de::GetLastError());

    VGT->run((_type_in*)src->Mat.ptr, (float2*)map->Mat.ptr, (_type_out*)dst->Mat.ptr, map->Pitch(), dst->Pitch(), &t1D);

    g_VGT2D_hdlr.unlock();
}


_DECX_API_ void 
de::dsp::cpu::Resample(de::InputMatrix src, de::InputMatrix map, de::OutputMatrix dst, de::Interpolate_Types interpoate_mode)
{
    de::ResetLastError();

    const decx::_Matrix* _src = dynamic_cast<const decx::_Matrix*>(&src);
    const decx::_Matrix* _map = dynamic_cast<const decx::_Matrix*>(&map);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!decx::cpu::_is_CPU_init()){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::resample_caller<float, float>(_src, _map, _dst, interpoate_mode, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::resample_caller<uint8_t, uint8_t>(_src, _map, _dst, interpoate_mode, de::GetLastError());
        break;
    
    default:
        break;
    }
}

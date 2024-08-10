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

#include "filter2D.h"
#include "2D/common/cpu_filter2D_planner.h"


namespace decx
{
namespace dsp{
    void filter2D_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* _dst, const de::extend_label padding,
        de::DH* handle);


    template <bool _cplxf>
    void filter2D_64b(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* _dst, const de::extend_label padding,
        de::DH* handle);
}
}


void decx::dsp::filter2D_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const de::extend_label padding,
    de::DH* handle)
{
    if (decx::dsp::g_cpu_filter2D_fp32._res_ptr == NULL){
        decx::dsp::g_cpu_filter2D_fp32.RegisterResource(new decx::dsp::cpu_Filter2D_planner<float>, 5,
            decx::dsp::cpu_Filter2D_planner<float>::release);
    }
    
    const uint2 dst_dims = decx::dsp::cpu_Filter2D_planner<void>::query_dst_dims(
        &src->get_layout(), &kernel->get_layout(), padding);

    dst->re_construct(de::_FP32_, dst_dims.x, dst_dims.y);

    decx::dsp::g_cpu_filter2D_fp32.lock();

    auto* _planner = decx::dsp::g_cpu_filter2D_fp32.get_resource_raw_ptr<decx::dsp::cpu_Filter2D_planner<float>>();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();

    if (_planner->changed(_conc, src, kernel, dst, padding)) {
        _planner->plan(_conc, &src->get_layout(), &kernel->get_layout(), &dst->get_layout(), de::GetLastError(), padding);
    }
    const uint2 thread_dist = _planner->get_thread_dist();
    decx::utils::_thr_2D t2D(thread_dist.y, thread_dist.x);
    _planner->run<false>(src, kernel, dst, &t2D);

    decx::dsp::g_cpu_filter2D_fp32.unlock();
}


template <bool _cplxf>
void decx::dsp::filter2D_64b(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const de::extend_label padding,
    de::DH* handle)
{
    if (decx::dsp::g_cpu_filter2D_64b._res_ptr == NULL) {
        decx::dsp::g_cpu_filter2D_64b.RegisterResource(new decx::dsp::cpu_Filter2D_planner<double>, 5,
            decx::dsp::cpu_Filter2D_planner<double>::release);
    }

    const uint2 dst_dims = decx::dsp::cpu_Filter2D_planner<void>::query_dst_dims(
        &src->get_layout(), &kernel->get_layout(), padding);

    dst->re_construct(src->Type(), dst_dims.x, dst_dims.y);

    decx::dsp::g_cpu_filter2D_64b.lock();

    auto* _planner = decx::dsp::g_cpu_filter2D_64b.get_resource_raw_ptr<decx::dsp::cpu_Filter2D_planner<double>>();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
    if (_planner->changed(_conc, src, kernel, dst, padding)) {
        _planner->plan(_conc, &src->get_layout(), &kernel->get_layout(), &dst->get_layout(), de::GetLastError(), padding);
    }
    const uint2 thread_dist = _planner->get_thread_dist();
    decx::utils::_thr_2D t2D(thread_dist.y, thread_dist.x);
    _planner->run<_cplxf>(src, kernel, dst, &t2D);

    decx::dsp::g_cpu_filter2D_64b.unlock();
}


_DECX_API_ void 
de::dsp::cpu::Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const de::extend_label padding_method, 
    const de::_DATA_TYPES_FLAGS_ output_type)
{
    de::ResetLastError();
    
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::filter2D_fp32(_src, _kernel, _dst, padding_method, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::filter2D_64b<false>(_src, _kernel, _dst, padding_method, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::filter2D_64b<true>(_src, _kernel, _dst, padding_method, de::GetLastError());
        break;
    default:
        break;
    }
}

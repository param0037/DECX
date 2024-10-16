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


#include "rotate_fp32.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../../common/FMGR/fragment_arrangment.h"


void 
decx::dsp::complex_rotate_fp32_caller(const double* src, const float angle, double* dst, const size_t _proc_len)
{
    decx::utils::_thread_arrange_1D t1D((uint)decx::cpu::_get_permitted_concurrency());

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, _proc_len, t1D.total_thread);

    de::CPf _rot_factor;
    _rot_factor.construct_with_phase(angle);

    const double* _loc_src = src;
    double* _loc_dst = dst;
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::cp_mul_c_fvec4_ST,
            _loc_src, *((double*)&_rot_factor), _loc_dst, f_mgr.frag_len);
        _loc_src += f_mgr.frag_len;
        _loc_dst += f_mgr.frag_len;
    }
    const size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::calc::CPUK::cp_mul_c_fvec4_ST,
        _loc_src, *((double*)&_rot_factor), _loc_dst, _L);

    t1D.__sync_all_threads();
}



_DECX_API_ de::DH Complex_Rotate(de::Matrix& src, const float angle, de::Matrix& dst)
{
    de::DH handle;
    
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const size_t _proc_len = (uint64_t)_src->Pitch() * (uint64_t)_src->Height() / 4;

    decx::dsp::complex_rotate_fp32_caller((double*)_src->Mat.ptr, angle, (double*)_dst->Mat.ptr, _proc_len);

    decx::err::Success(&handle);

    return handle;
}


_DECX_API_ de::DH Complex_Rotate(de::Vector& src, const float angle, de::Vector& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    const size_t _proc_len = _src->_length / 4;

    decx::dsp::complex_rotate_fp32_caller((double*)_src->Vec.ptr, angle, (double*)_dst->Vec.ptr, _proc_len);

    decx::err::Success(&handle);

    return handle;
}
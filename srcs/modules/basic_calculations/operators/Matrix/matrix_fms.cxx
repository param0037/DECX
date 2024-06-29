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


#include "../Fms_exec.h"
#include "../../operators_frame_exec.h"
#include "../../../core/configs/config.h"
#include "Matrix_operators.h"



_DECX_API_ de::DH de::cpu::Fms(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    if (!(_A->is_init() && _B->is_init() && _C->is_init())) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_m3<decx::calc::_fp32_ternary_ops_m, float, 8>(decx::calc::CPUK::fms_m_fvec8_ST,
            (float*)_A->Mat.ptr, (float*)_B->Mat.ptr, (float*)_C->Mat.ptr, (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_m3<decx::calc::_int32_ternary_ops_m, int, 8>(decx::calc::CPUK::fms_m_ivec8_ST,
            (int*)_A->Mat.ptr, (int*)_B->Mat.ptr, (int*)_C->Mat.ptr, (int*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 4, t1D->total_thread);

        decx::calc::operators_caller_m3<decx::calc::_fp64_ternary_ops_m, double, 4>(decx::calc::CPUK::fms_m_dvec4_ST,
            (double*)_A->Mat.ptr, (double*)_B->Mat.ptr, (double*)_C->Mat.ptr, (double*)_dst->Mat.ptr, t1D, &f_mgr);
        break;
    default:
        break;
    }

    if (t1D != NULL) {
        delete t1D;
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::cpu::Fms(de::Matrix& A, void* __x, de::Matrix& B, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (__x == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_c3<decx::calc::_fp32_ternary_ops_c, float, 8>(decx::calc::CPUK::fms_c_fvec8_ST,
            (float*)_A->Mat.ptr, *((float*)__x), (float*)_B->Mat.ptr, (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_c3<decx::calc::_int32_ternary_ops_c, int, 8>(decx::calc::CPUK::fms_c_ivec8_ST,
            (int*)_A->Mat.ptr, *((int*)__x), (int*)_B->Mat.ptr, (int*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 4, t1D->total_thread);

        decx::calc::operators_caller_c3<decx::calc::_fp64_ternary_ops_c, double, 8>(decx::calc::CPUK::fms_c_dvec4_ST,
            (double*)_A->Mat.ptr, *((double*)__x), (double*)_B->Mat.ptr, (double*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    default:
        break;
    }

    if (t1D != NULL) {
        delete t1D;
    }

    decx::err::Success(&handle);
    return handle;
}
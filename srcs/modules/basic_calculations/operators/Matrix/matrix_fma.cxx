/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../Fma_exec.h"
#include "../../operators_frame_exec.h"
#include "../../../core/configs/config.h"
#include "Matrix_operators.h"



_DECX_API_ de::DH de::cpu::Fma(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!(_A->is_init() && _B->is_init() && _C->is_init())) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_m3<decx::calc::_fp32_ternary_ops_m, float, 8>(decx::calc::CPUK::fma_m_fvec8_ST,
            (float*)_A->Mat.ptr, (float*)_B->Mat.ptr, (float*)_C->Mat.ptr, (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_m3<decx::calc::_int32_ternary_ops_m, int, 8>(decx::calc::CPUK::fma_m_ivec8_ST,
            (int*)_A->Mat.ptr, (int*)_B->Mat.ptr, (int*)_C->Mat.ptr, (int*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 4, t1D->total_thread);

        decx::calc::operators_caller_m3<decx::calc::_fp64_ternary_ops_m, double, 4>(decx::calc::CPUK::fma_m_dvec4_ST,
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



_DECX_API_ de::DH de::cpu::Fma(de::Matrix& A, void* __x, de::Matrix& B, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (__x == NULL) {
        Print_Error_Message(4, "Can't find entity of parameter '__x'\n");
        decx::err::InvalidParam(&handle);
        return handle;
    }

    if (!(_A->is_init() && _B->is_init())) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_c3<decx::calc::_fp32_ternary_ops_c, float, 8>(decx::calc::CPUK::fma_c_fvec8_ST,
            (float*)_A->Mat.ptr, *((float*)__x), (float*)_B->Mat.ptr, (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_c3<decx::calc::_int32_ternary_ops_c, int, 8>(decx::calc::CPUK::fma_c_ivec8_ST,
            (int*)_A->Mat.ptr, *((int*)__x), (int*)_B->Mat.ptr, (int*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->Pitch() * _A->Height() / 4, t1D->total_thread);

        decx::calc::operators_caller_c3<decx::calc::_fp64_ternary_ops_c, double, 8>(decx::calc::CPUK::fma_c_dvec4_ST,
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
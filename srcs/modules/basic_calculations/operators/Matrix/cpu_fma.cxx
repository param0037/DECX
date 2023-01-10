/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "../Fma_exec.h"
#include "../operators_frame_exec.h"
#include "../../../classes/Matrix.h"
#include "../../../core/configs/config.h"


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Fma(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


        _DECX_API_ de::DH Fma(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);
    }
}


_DECX_API_ de::DH de::cpu::Fma(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpI.cpu_concurrency);
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_element_num / 8, t1D->total_thread);
        decx::calc::operators_caller_m_fp32(decx::calc::CPUK::fma_m_fvec8_ST,
            (float*)_A->Mat.ptr, (float*)_B->Mat.ptr, (float*)_C->Mat.ptr, (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpI.cpu_concurrency);
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_element_num / 8, t1D->total_thread);
        decx::calc::operators_caller_m_int32(decx::calc::CPUK::fma_m_ivec8_ST,
            (int*)_A->Mat.ptr, (int*)_B->Mat.ptr, (int*)_C->Mat.ptr, (int*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpI.cpu_concurrency);
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_element_num / 4, t1D->total_thread);
        decx::calc::operators_caller_m_fp64(decx::calc::CPUK::fma_m_dvec4_ST,
            (double*)_A->Mat.ptr, (double*)_B->Mat.ptr, (double*)_C->Mat.ptr, (double*)_dst->Mat.ptr, t1D, &f_mgr);
        break;
    default:
        break;
    }

    if (t1D != NULL) {
        delete t1D;
    }

    decx::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::cpu::Fma(de::Matrix& A, void* __x, de::Matrix& B, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
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

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpI.cpu_concurrency);
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_element_num / 8, t1D->total_thread);
        decx::calc::operators_caller_c_fp32(decx::calc::CPUK::fma_c_fvec8_ST,
            (float*)_A->Mat.ptr, *((float*)__x), (float*)_B->Mat.ptr, (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpI.cpu_concurrency);
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_element_num / 8, t1D->total_thread);
        decx::calc::operators_caller_c_int32(decx::calc::CPUK::fma_c_ivec8_ST,
            (int*)_A->Mat.ptr, *((int*)__x), (int*)_B->Mat.ptr, (int*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpI.cpu_concurrency);
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_element_num / 4, t1D->total_thread);
        decx::calc::operators_caller_c_fp64(decx::calc::CPUK::fma_c_dvec4_ST,
            (double*)_A->Mat.ptr, *((double*)__x), (double*)_B->Mat.ptr, (double*)_dst->Mat.ptr, t1D, &f_mgr);
        break;
    default:
        break;
    }

    if (t1D != NULL) {
        delete t1D;
    }

    decx::Success(&handle);
    return handle;
}
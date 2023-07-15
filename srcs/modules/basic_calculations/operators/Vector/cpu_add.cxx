/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../Add_exec.h"
#include "../../operators_frame_exec.h"
#include "../../../classes/Vector.h"
#include "../../../core/configs/config.h"
#include "../cp_ops_exec.h"


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Add(de::Vector& A, de::Vector& B, de::Vector& dst);


        _DECX_API_ de::DH Add(de::Vector& src, void* __x, de::Vector& dst);
    }
}


_DECX_API_ de::DH de::cpu::Add(de::Vector& A, de::Vector& B, de::Vector& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!(_A->is_init() && _B->is_init())) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 8, t1D->total_thread);

        decx::calc::operators_caller_m<decx::calc::_fp32_binary_ops_m, float, 8>(decx::calc::CPUK::add_m_fvec8_ST,
            (float*)_A->Vec.ptr, (float*)_B->Vec.ptr, (float*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 8, t1D->total_thread);

        decx::calc::operators_caller_m<decx::calc::_int32_binary_ops_m, int, 8>(decx::calc::CPUK::add_m_ivec8_ST,
            (int*)_A->Vec.ptr, (int*)_B->Vec.ptr, (int*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 4, t1D->total_thread);

        decx::calc::operators_caller_m<decx::calc::_fp64_binary_ops_m, double, 4>(decx::calc::CPUK::add_m_dvec4_ST,
            (double*)_A->Vec.ptr, (double*)_B->Vec.ptr, (double*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 4, t1D->total_thread);

        decx::calc::operators_caller_m<decx::calc::_fp64_binary_ops_m, double, 4>(decx::calc::CPUK::cp_add_m_fvec4_ST,
            (double*)_A->Vec.ptr, (double*)_B->Vec.ptr, (double*)_dst->Vec.ptr, t1D, &f_mgr);
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



_DECX_API_ de::DH de::cpu::Add(de::Vector& src, void* __x, de::Vector& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (__x == NULL) {
        Print_Error_Message(4, "Can't find entity of parameter '__x'\n");
        decx::err::InvalidParam(&handle);
        return handle;
    }

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_fp32_binary_ops_c, float, 8>(decx::calc::CPUK::add_c_fvec8_ST,
            (float*)_src->Vec.ptr, *((float*)__x), (float*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_int32_binary_ops_c, int, 8>(decx::calc::CPUK::add_c_ivec8_ST,
            (int*)_src->Vec.ptr, *((int*)__x), (int*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 4, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_fp64_binary_ops_c, double, 8>(decx::calc::CPUK::cp_add_c_fvec4_ST,
            (double*)_src->Vec.ptr, *((double*)__x), (double*)_dst->Vec.ptr, t1D, &f_mgr);
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
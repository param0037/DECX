/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../Div_exec.h"
#include "../../operators_frame_exec.h"
#include "../../../classes/Vector.h"
#include "../../../core/configs/config.h"


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Div(de::Vector& A, de::Vector& B, de::Vector& dst);


        _DECX_API_ de::DH Div(de::Vector& src, void* __x, de::Vector& dst);


        _DECX_API_ de::DH Div(void* __x, de::Vector& src, de::Vector& dst);
    }
}


_DECX_API_ de::DH de::cpu::Div(de::Vector& A, de::Vector& B, de::Vector& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!(_A->is_init() && _B->is_init())) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_A->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_m<decx::calc::_fp32_binary_ops_m, float, 8>(decx::calc::CPUK::div_m_fvec8_ST,
            (float*)_A->Vec.ptr, (float*)_B->Vec.ptr, (float*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_m<decx::calc::_int32_binary_ops_m, int, 8>(decx::calc::CPUK::div_m_ivec8_ST,
            (int*)_A->Vec.ptr, (int*)_B->Vec.ptr, (int*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        // configure the process domain for each thread
        decx::utils::frag_manager_gen(&f_mgr, _A->_length / 4, t1D->total_thread);
        
        decx::calc::operators_caller_m<decx::calc::_fp64_binary_ops_m, double, 4>(decx::calc::CPUK::div_m_dvec4_ST,
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



_DECX_API_ de::DH de::cpu::Div(de::Vector& src, void* __x, de::Vector& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (__x == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_fp32_binary_ops_c, float, 8>(decx::calc::CPUK::div_c_fvec8_ST,
            (float*)_src->Vec.ptr, *((float*)__x), (float*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_int32_binary_ops_c, int, 8>(decx::calc::CPUK::div_c_ivec8_ST,
            (int*)_src->Vec.ptr, *((int*)__x), (int*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 4, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_fp64_binary_ops_c, double, 8>(decx::calc::CPUK::div_c_dvec4_ST,
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



_DECX_API_ de::DH de::cpu::Div(void* __x, de::Vector& src, de::Vector& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (__x == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_fp32_binary_ops_c, float, 8>(decx::calc::CPUK::div_cinv_fvec8_ST,
            (float*)_src->Vec.ptr, *((float*)__x), (float*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 8, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_int32_binary_ops_c, int, 8>(decx::calc::CPUK::div_cinv_ivec8_ST,
            (int*)_src->Vec.ptr, *((int*)__x), (int*)_dst->Vec.ptr, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager_gen(&f_mgr, _src->_length / 4, t1D->total_thread);
        
        decx::calc::operators_caller_c<decx::calc::_fp64_binary_ops_c, double, 8>(decx::calc::CPUK::div_cinv_dvec4_ST,
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
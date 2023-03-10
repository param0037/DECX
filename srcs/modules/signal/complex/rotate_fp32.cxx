/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "rotate_fp32.h"


void 
decx::signal::complex_rotate_fp32_caller(const double* src, const float angle, double* dst, const size_t _proc_len)
{
    decx::utils::_thread_arrange_1D t1D((uint)decx::cpI.cpu_concurrency);

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, _proc_len, t1D.total_thread);

    de::CPf _rot_factor;
    _rot_factor.construct_with_phase(angle);

    const double* _loc_src = src;
    double* _loc_dst = dst;
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::calc::CPUK::cp_mul_c_fvec4_ST,
            _loc_src, *((double*)&_rot_factor), _loc_dst, f_mgr.frag_len);
        _loc_src += f_mgr.frag_len;
        _loc_dst += f_mgr.frag_len;
    }
    const size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::calc::CPUK::cp_mul_c_fvec4_ST,
        _loc_src, *((double*)&_rot_factor), _loc_dst, _L);

    t1D.__sync_all_threads();
}



_DECX_API_ de::DH Complex_Rotate(de::Matrix& src, const float angle, de::Matrix& dst)
{
    de::DH handle;
    
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const size_t _proc_len = _src->_element_num / 4;

    decx::signal::complex_rotate_fp32_caller((double*)_src->Mat.ptr, angle, (double*)_dst->Mat.ptr, _proc_len);

    decx::err::Success(&handle);

    return handle;
}


_DECX_API_ de::DH Complex_Rotate(de::Vector& src, const float angle, de::Vector& dst)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    const size_t _proc_len = _src->_length / 4;

    decx::signal::complex_rotate_fp32_caller((double*)_src->Vec.ptr, angle, (double*)_dst->Vec.ptr, _proc_len);

    decx::err::Success(&handle);

    return handle;
}
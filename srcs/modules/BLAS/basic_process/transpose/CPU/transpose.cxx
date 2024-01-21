/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "transpose.h"


_DECX_API_ de::DH de::cpu::Transpose(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (_src->get_layout()._single_element_size == sizeof(float)) 
    {
        decx::bp::_cpu_transpose_config<4> config(make_uint2(_src->Width(), _src->Height()), t1D.total_thread);

        decx::bp::transpose_4x4_caller((const float*)_src->Mat.ptr, (float*)_dst->Mat.ptr,
            _src->Pitch(), _dst->Pitch(), &config, &t1D);
    }
    else if (_src->get_layout()._single_element_size == sizeof(uint8_t)) {
        decx::bp::_cpu_transpose_config<1> config(make_uint2(_src->Width(), _src->Height()), t1D.total_thread);

        decx::bp::transpose_8x8_caller((const double*)_src->Mat.ptr, (double*)_dst->Mat.ptr,
            _src->Pitch(), _dst->Pitch(), &config, &t1D);
    }
    else if (_src->get_layout()._single_element_size == sizeof(double)) 
    {
        decx::bp::_cpu_transpose_config<8> config(make_uint2(_src->Width(), _src->Height()), t1D.total_thread);

        decx::bp::transpose_2x2_caller((const double*)_src->Mat.ptr, (double*)_dst->Mat.ptr, 
             _src->Pitch(), _dst->Pitch(), &config, &t1D);
    }

    return handle;
}
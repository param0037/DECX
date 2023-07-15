/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "histogram.h"
#include "hist_gen_exec.h"


_DECX_API_ de::DH de::cpu::Histogram(de::Matrix& src, de::Vector& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::CPU_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_src->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_UINT8_:
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_UINT64_, 256, _src->get_store_type());
        decx::bp::_histgen2D_u8_caller((uint8_t*)_src->Mat.ptr, (uint64_t*)_dst->Vec.ptr, 
            make_uint2(_src->Width(), _src->Height()), _src->Pitch());
        break;
    default:
        break;
    }
    
    decx::err::Success(&handle);
    return handle;
}
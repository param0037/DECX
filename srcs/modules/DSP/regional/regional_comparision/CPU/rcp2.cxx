/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "fp32/rcp2_SQDIFF_fp32.h"
#include "fp32/rcp2_CCOEFF_fp32.h"
#include "uint8/rcp2_SQDIFF_uint8.h"
#include "uint8/rcp2_CCOEFF_uint8.h"
#include "../rcp_flags.h"


namespace decx
{
    namespace rcp {

        void rcp2_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int calc_flag, de::DH* handle);
                                                                                                           
                         

        void rcp2_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int calc_flag, de::DH* handle);
    }
}



void decx::rcp::rcp2_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int calc_flag, de::DH* handle)
{
    dst->re_construct(src->Type(), src->Width() - kernel->Width() + 1, src->Height() - kernel->Height() + 1);
    switch (calc_flag)
    {
    case decx::rcp::_RCP_FLAGS_::RCP_SQDIFF:
        decx::rcp::_rcp2_SQDIFF_fp32_NB<false>(src, kernel, dst, handle);
        break;
    case decx::rcp::_RCP_FLAGS_::RCP_SQDIFF_NORMAL:
        decx::rcp::_rcp2_SQDIFF_fp32_NB<true>(src, kernel, dst, handle);
        break;

    case decx::rcp::_RCP_FLAGS_::RCP_CCOEFF:
        decx::rcp::_rcp2_CCOEFF_fp32_NB<false>(src, kernel, dst, handle);
        break;
    case decx::rcp::_RCP_FLAGS_::RCP_CCOEFF_NORMAL:
        decx::rcp::_rcp2_CCOEFF_fp32_NB<true>(src, kernel, dst, handle);
        break;
    default:
        break;
    }
}


void decx::rcp::rcp2_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int calc_flag, de::DH* handle)
{
    switch (calc_flag)
    {
    case decx::rcp::_RCP_FLAGS_::RCP_SQDIFF:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_INT32_, src->Width() - kernel->Width() + 1, 
                      src->Height() - kernel->Height() + 1);
        decx::rcp::_rcp2_SQDIFF_uint8<false>(src, kernel, dst, handle);
        break;
    case decx::rcp::_RCP_FLAGS_::RCP_SQDIFF_NORMAL:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width() - kernel->Width() + 1, 
                      src->Height() - kernel->Height() + 1);
        decx::rcp::_rcp2_SQDIFF_uint8<true>(src, kernel, dst, handle);
        break;

    case decx::rcp::_RCP_FLAGS_::RCP_CCOEFF:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1);
        decx::rcp::_rcp2_CCOEFF_uint8<false>(src, kernel, dst, handle);
        break;

    case decx::rcp::_RCP_FLAGS_::RCP_CCOEFF_NORMAL:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1);
        decx::rcp::_rcp2_CCOEFF_uint8<true>(src, kernel, dst, handle);
        break;

    default:
        break;
    }
}


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Regional_Comparision(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int calc_flag);
    }
}


_DECX_API_
de::DH de::cpu::Regional_Comparision(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int calc_flag)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    _dst->re_construct(_src->Type(), _src->Width() - _kernel->Width() + 1, _src->Height() - _kernel->Height() + 1);
    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::rcp::rcp2_fp32(_src, _kernel, _dst, calc_flag, &handle);
        break;
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::rcp::rcp2_uint8(_src, _kernel, _dst, calc_flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}
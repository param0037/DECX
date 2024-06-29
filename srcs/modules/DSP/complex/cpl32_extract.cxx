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


#include "cpl32_extract.h"
#include "../../handles/decx_handles.h"


_DECX_API_ de::DH de::dsp::cpu::Module(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->Type() != de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::dsp::_cpl32_extract_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Pitch() / 4, _src->Height()),
        _src->Pitch(), _dst->Pitch(), decx::dsp::CPUK::_module_fp32_ST2D);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::dsp::cpu::Angle(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->Type() != de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::dsp::_cpl32_extract_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Pitch() / 4, _src->Height()),
        _src->Pitch(), _dst->Pitch(), decx::dsp::CPUK::_angle_fp32_ST2D);

    decx::err::Success(&handle);
    return handle;
}
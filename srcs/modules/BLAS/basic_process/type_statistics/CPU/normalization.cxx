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


#include "normalization.h"


_THREAD_FUNCTION_ void 
decx::bp::CPUK::normalize_scale_v8_fp32(const float* src, float* dst, const double2 min_max, const double2 range, const uint64_t proc_len)
{
    const __m256 min_max_dst = _mm256_set1_ps(min_max.y - min_max.x);
    const __m256 range_dst = _mm256_set1_ps(range.y - range.x);
    __m256 recv, store;

    for (uint32_t i = 0; i < proc_len; ++i) {
        recv = _mm256_load_ps(src + (i << 3));

        recv = _mm256_sub_ps(recv, _mm256_set1_ps(min_max.x));
        recv = _mm256_div_ps(recv, min_max_dst);
        store = _mm256_fmadd_ps(recv, range_dst, _mm256_set1_ps(range.x));

        _mm256_store_ps(dst + (i << 3), store);
    }
}


_DECX_API_ void 
decx::cpu::scale_raw_API(decx::_Vector* src, decx::_Vector* dst, const double2 range, de::DH* handle)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_)
    {
        float _min_val = 0, _max_val = 0;
        decx::bp::_min_max_1D_caller<decx::bp::CPUK::_bicmp_kernel_fp32_1D, float, 8>(
            decx::bp::CPUK::_min_max_vec8_fp32_1D, (float*)src->Vec.ptr,
            src->_length, &_min_val, &_max_val);

        decx::bp::norm_caller<decx::bp::CPUK::norm_scale_kernel_fp32, float, 8>(
            decx::bp::CPUK::normalize_scale_v8_fp32, (float*)src->Vec.ptr, (float*)dst->Vec.ptr,
            make_double2(_min_val, _max_val), range, src->_length);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_)
    {

    }
}



_DECX_API_ void
decx::cpu::scale_raw_API(decx::_Matrix* src, decx::_Matrix* dst, const double2 range, de::DH* handle)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_)
    {
        float _min_val = 0, _max_val = 0;
        decx::bp::_min_max_2D_caller<decx::bp::CPUK::_bicmp_kernel_fp32_2D, float, 8>(
            decx::bp::CPUK::_min_max_vec8_fp32_2D, (float*)src->Mat.ptr,
            make_uint2(src->Width(), src->Height()), src->Pitch(), &_min_val, &_max_val);

        decx::bp::norm_caller<decx::bp::CPUK::norm_scale_kernel_fp32, float, 8>(
            decx::bp::CPUK::normalize_scale_v8_fp32, (float*)src->Mat.ptr, (float*)dst->Mat.ptr, 
            make_double2(_min_val, _max_val), range, 
            (uint64_t)src->Pitch() * (uint64_t)src->Height());
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_)
    {

    }
}


_DECX_API_ de::DH de::cpu::Scale(de::Vector& src, de::Vector& dst, de::Point2D_d range)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    decx::cpu::scale_raw_API(_src, _dst, make_double2(range.x, range.y), &handle);

    return handle;
}



_DECX_API_ de::DH de::cpu::Scale(de::Matrix& src, de::Matrix& dst, de::Point2D_d range)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::cpu::scale_raw_API(_src, _dst, make_double2(range.x, range.y), &handle);

    return handle;
}
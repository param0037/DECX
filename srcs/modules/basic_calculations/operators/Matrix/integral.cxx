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


#include "integral.h"


_THREAD_FUNCTION_ void
decx::calc::CPUK::_integral_H_fp32_2D_ST(const float* __restrict src, 
                                         float* __restrict dst, 
                                         const uint2 proc_dims, 
                                         const uint pitchsrc, 
                                         const uint pitchdst)
{
    size_t dex_src, dex_dst;
    float buffer = 0, reg = 0;
    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = (size_t)i * (size_t)pitchsrc;
        dex_dst = (size_t)i * (size_t)pitchdst;
        buffer = src[dex_src];
        dst[dex_dst] = buffer;
        ++dex_src;
        ++dex_dst;
        for (int j = 1; j < proc_dims.x; ++j) {
            reg = src[dex_src];
            buffer += reg;
            dst[dex_dst] = buffer;
            ++dex_src;
            ++dex_dst;
        }
    }
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::_integral_V_fp32_2D_ST(float* __restrict src, 
                                         const uint2 proc_dims,     // ~.x -> in __m256
                                         const uint pitch)          // in float
{
    size_t dex;
    __m256 buffer = _mm256_set1_ps(0), reg = _mm256_set1_ps(0);
    for (int i = 0; i < proc_dims.x; ++i) {
        dex = ((size_t)i << 3);
        buffer = _mm256_load_ps(src + dex);
        dex += (size_t)pitch;
        for (int j = 1; j < proc_dims.y; ++j) {
            reg = _mm256_load_ps(src + dex);
            buffer = _mm256_add_ps(reg, buffer);
            _mm256_store_ps(src + dex, buffer);
            dex += (size_t)pitch;
        }
    }
}



_DECX_API_
void decx::calc::_integral_caller2D_fp32(const float *src, float* dst, const uint2 proc_dims,
    const uint pitchsrc, const uint pitchdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgrH, f_mgrV;
    decx::utils::frag_manager_gen(&f_mgrH, proc_dims.y, t1D.total_thread);
    decx::utils::frag_manager_gen(&f_mgrV, pitchdst / 8, t1D.total_thread);

    const float* _thr_src_ptr = src;
    float* _thr_dst_ptr = dst;
    const size_t frag_src = (size_t)pitchsrc * (size_t)f_mgrH.frag_len;
    const size_t frag_dst = (size_t)pitchdst * (size_t)f_mgrH.frag_len;
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_H_fp32_2D_ST,
            _thr_src_ptr, _thr_dst_ptr, make_uint2(proc_dims.x, f_mgrH.frag_len), pitchsrc, pitchdst);
        _thr_src_ptr += frag_src;
        _thr_dst_ptr += frag_dst;
    }
    uint2 proc_dim_left = make_uint2(proc_dims.x, f_mgrH.is_left ? f_mgrH.frag_left_over : f_mgrH.frag_len);
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_H_fp32_2D_ST,
        _thr_src_ptr, _thr_dst_ptr, proc_dim_left, pitchsrc, pitchdst);

    t1D.__sync_all_threads();

    _thr_dst_ptr = dst;

    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_V_fp32_2D_ST,
            _thr_dst_ptr, make_uint2(f_mgrV.frag_len, proc_dims.y), pitchdst);
        _thr_dst_ptr += f_mgrV.frag_len * 8;
    }
    proc_dim_left = make_uint2(f_mgrV.is_left ? f_mgrV.frag_left_over : f_mgrV.frag_len, proc_dims.y);
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_V_fp32_2D_ST,
        _thr_dst_ptr, proc_dim_left, pitchdst);

    t1D.__sync_all_threads();
}




_DECX_API_ de::DH de::calc::cpu::Integral(de::Matrix& src, de::Matrix& dst)
{
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::_integral_caller2D_fp32(
            (float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Width(), _src->Height()), _src->Pitch(), _dst->Pitch());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::calc::_integral_caller2D_uint8((uint8_t*)_src->Mat.ptr, (int32_t*)_dst->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), _dst->Pitch());
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "VEVID.h"
#include "../../../BLAS/basic_process/type_statistics/CPU/cmp_exec.h"
#include "../../../basic_calculations/operators/Maprange_exec.h"
#include "../../../core/utils/simd_fast_math_avx2.h"


_THREAD_FUNCTION_ void 
decx::vis::CPUK::_VEVID_u8_kernel(const double* __restrict src, 
                                  float* __restrict dst, 
                                  const uint32_t pitchsrc_v8, 
                                  const uint32_t pitchdst_v1, 
                                  const uint32_t proc_H,
                                  const float _phase_gain,
                                  const float _original_gain)
{
    double _pixels_IO_v8;
    decx::utils::simd::xmm256_reg _reg;

    uint64_t dex_src = 0, dex_dst = 0;;

    for (uint32_t i = 0; i < proc_H; ++i)
    {
        dex_src = i * pitchsrc_v8;
        dex_dst = i * pitchdst_v1;
        for (uint32_t j = 0; j < min(pitchsrc_v8, pitchdst_v1 / 8); ++j) {
            _pixels_IO_v8 = src[dex_src];
            _reg._arrd[0] = _pixels_IO_v8;
            _reg._vi = _mm256_permutevar8x32_epi32(_reg._vi, _mm256_setr_epi32(0, 2, 2, 2, 1, 5, 5, 5));
            _reg._vi = _mm256_and_si256(_mm256_shuffle_epi8(_reg._vi, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)), _mm256_set1_epi32(0xff));
            _reg._vf = _mm256_cvtepi32_ps(_reg._vi);

            _reg._vf = _mm256_div_ps(_mm256_set1_ps(_original_gain), _reg._vf);
            _reg._vf = _mm256_mul_ps(_mm256_set1_ps(-_phase_gain), _mm256_add_ps(_reg._vf, _mm256_set1_ps(1.f)));
            
            _reg._vf = decx::utils::simd::_mm256_atan_ps(_reg._vf);

            _mm256_store_ps(dst + dex_dst, _reg._vf);

            ++dex_src;
            dex_dst += 8;
        }
    }
}


void decx::vis::VEVID_u8_caller(const double* src, 
                                float* dst, 
                                const uint32_t pitchsrc_v8, 
                                const uint32_t pitchdst_v1, 
                                const uint32_t proc_H, 
                                const float _phase_gain,
                                const float _original_gain)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_H, t1D.total_thread);

    const double* _loc_src = src;
    float* _loc_dst = dst;
    for (uint32_t i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::vis::CPUK::_VEVID_u8_kernel,
            _loc_src, _loc_dst, pitchsrc_v8, pitchdst_v1, f_mgr.frag_len, _phase_gain, _original_gain);

        _loc_src += pitchsrc_v8 * f_mgr.frag_len;
        _loc_dst += pitchdst_v1 * f_mgr.frag_len;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(decx::vis::CPUK::_VEVID_u8_kernel,
        _loc_src, _loc_dst, pitchsrc_v8, pitchdst_v1, _L, _phase_gain, _original_gain);

    t1D.__sync_all_threads();
}


_DECX_API_ de::DH de::vis::cpu::VEVID_gray(de::Matrix& src, de::Matrix& dst, const float _phase_gain,
    const float _original_gain)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const uint2 _tmp_dims = make_uint2(decx::utils::align<uint32_t>(_src->Width(), 8), _src->Height());
    decx::PtrInfo<float> _tmp;
    if (decx::alloc::_host_virtual_page_malloc(&_tmp, _tmp_dims.x * _tmp_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return handle;
    }

    decx::vis::VEVID_u8_caller((double*)_src->Mat.ptr, _tmp.ptr, _src->Pitch() / 8, _tmp_dims.x,
        _src->Height(), _phase_gain, _original_gain);

    float _max = NULL, _min = NULL;
    decx::bp::_min_max_2D_caller<decx::bp::CPUK::_bicmp_kernel_fp32_2D, 
                                float, 8>(decx::bp::CPUK::_min_max_vec8_fp32_2D,
                                          _tmp.ptr, make_uint2(_src->Width(), _src->Height()), 
                                          _tmp_dims.x, &_max, &_min);

    decx::calc::maprange2D_cvtf32_u8_caller(_tmp.ptr, (double*)_dst->Mat.ptr, _tmp_dims.x, _dst->Pitch() / 8, 
        make_uint2(_src->Width(), _src->Height()), make_float2(_min, _max));

    decx::alloc::_host_virtual_page_dealloc(&_tmp);

    return handle;
}
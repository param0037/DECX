/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "cvt_colors.h"
#include "../../core/utils/fragment_arrangment.h"


_THREAD_FUNCTION_ void 
decx::vis::_BGR2Gray_ST_UC2UC(const float* __restrict    src, 
                              float* __restrict          dst, 
                              const int2                 dims,
                              const uint                 pitchsrc, 
                              const uint                 pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m128i __recv;
    __m128i __res, __buf;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __recv = _mm_shuffle_epi8(__recv, _shuffle_var);

            __buf = _mm_cvtepu8_epi32(__recv);
            __res = _mm_mullo_epi32(__buf, _mm_set1_epi32(19595));

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b01010101));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(38469));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b10101010));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(7472));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_srli_epi32(__res, 16);
            __res = _mm_shuffle_epi8(__buf, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            
            ++glo_dex_dst;
        }
    }
}


_THREAD_FUNCTION_ void 
decx::vis::_BGR2Mean_ST_UC2UC(const float* __restrict   src, 
                              float* __restrict         dst, 
                              const int2                dims, 
                              const uint                pitchsrc, 
                              const uint                pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m128i __recv;
    __m128i __res, __buf;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __recv = _mm_shuffle_epi8(__recv, _shuffle_var);

            __buf = _mm_cvtepu8_epi32(__recv);
            __res = _mm_mullo_epi32(__buf, _mm_set1_epi32(21846));

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b01010101));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(21846));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b10101010));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(21846));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_srli_epi32(__res, 16);
            __res = _mm_shuffle_epi8(__buf, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::_Preserve_B_ST_UC2UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint                  pitchsrc, 
                                const uint                  pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::_Preserve_G_ST_UC2UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint                  pitchsrc, 
                                const uint                  pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::_Preserve_R_ST_UC2UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint                  pitchsrc, 
                                const uint                  pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::_Preserve_A_ST_UC2UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint                  pitchsrc, 
                                const uint                  pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(3, 7, 11, 15, 1, 5, 9, 13, 0, 4, 8, 12, 2, 6, 10, 14);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}


// --------------------------------------- CALLERS --------------------------------------------------------


void decx::vis::_channel_ops_general_caller(decx::vis::channel_ops_kernel kernel, const float* src, float* dst, const int2 dims, 
    const uint pitchsrc, const uint pitchdst)
{
    int _concurrent = (int)decx::cpu::_get_permitted_concurrency();
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment_src = pitchsrc * (size_t)sub_dims.y, 
        fragment_dst = pitchdst * (size_t)sub_dims.y / 4,
        offset_src = 0,
        offset_dst = 0;

    std::future<void>* _thread_handle = new std::future<void>[_concurrent];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::cpu::register_task_default(kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);
        offset_src += fragment_src;
        offset_dst += fragment_dst;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::cpu::_get_permitted_concurrency() - 1] =
        decx::cpu::register_task_default(kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}

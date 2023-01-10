/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "template_configurations.h"


void
decx::rcp::CPUK::_template_sq_sum_vec8_fp32(const float* src, const size_t len, float* res_vec)
{
    __m256 tmp_recv, sum_vec8 = _mm256_set1_ps(0);

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_ps(src + ((size_t)i << 3));
        sum_vec8 = _mm256_fmadd_ps(tmp_recv, tmp_recv, sum_vec8);
    }

    *res_vec = decx::utils::simd::_mm256_h_sum(sum_vec8);
}


void
decx::rcp::CPUK::_template_sq_sum_vec8_uint8(const uint8_t* src, const size_t len, float* res_vec)
{
    uint8_t recv_uint8[8];
    __m256i tmp_recv, sum_vec8 = _mm256_set1_epi32(0);

    for (uint i = 0; i < len; ++i) {
        *((double*)recv_uint8) = src[(size_t)i << 3];
        tmp_recv = _mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_load_pd((double*)recv_uint8)));
        sum_vec8 = _mm256_add_epi32(_mm256_mul_epi32(tmp_recv, tmp_recv), sum_vec8);
    }
    sum_vec8 = _mm256_castps_si256(_mm256_cvtepi32_ps(sum_vec8));
    *res_vec = decx::utils::simd::_mm256_h_sum(_mm256_castsi256_ps(sum_vec8));
}



void
decx::rcp::CPUK::_template_sum_vec8_fp32(const float* src, const size_t len, float* res_vec)
{
    __m256 tmp_recv, sum_vec8 = _mm256_set1_ps(0);

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_ps(src + ((size_t)i << 3));
        sum_vec8 = _mm256_add_ps(tmp_recv, sum_vec8);
    }

    *res_vec = decx::utils::simd::_mm256_h_sum(sum_vec8);
}


void
decx::rcp::CPUK::_template_sum_vec4_fp64(const double* src, const size_t len, double* res_vec)
{
    __m256d tmp_recv, sum_vec8 = _mm256_set1_pd(0);

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_pd(src + ((size_t)i << 2));
        sum_vec8 = _mm256_add_pd(tmp_recv, sum_vec8);
    }

    *res_vec = decx::utils::simd::_mm256d_h_sum(sum_vec8);
}



void
decx::rcp::CPUK::_template_normalize_fp32(const float* src, float* dst, const size_t len, const uint2 actual_dims)
{
    float sum = 0;
    decx::rcp::CPUK::_template_sum_vec8_fp32(src, len, &sum);

    __m256 _buffer, _sub = _mm256_set1_ps((float)actual_dims.x * (float)actual_dims.y / sum);
    
    for (uint i = 0; i < len; ++i) {
        _buffer = _mm256_load_ps(src + ((size_t)i << 3));
        _buffer = _mm256_sub_ps(_buffer, _sub);
        _mm256_store_ps(dst + ((size_t)i << 3), _buffer);
    }
}



void
decx::rcp::CPUK::_template_normalize_fp64(const double* src, double* dst, const size_t len, const uint2 actual_dims)
{
    double sum = 0;
    decx::rcp::CPUK::_template_sum_vec4_fp64(src, len, &sum);

    __m256d _buffer, _sub = _mm256_set1_pd((float)actual_dims.x * (float)actual_dims.y / sum);

    for (uint i = 0; i < len; ++i) {
        _buffer = _mm256_load_pd(src + ((size_t)i << 2));
        _buffer = _mm256_sub_pd(_buffer, _sub);
        _mm256_store_pd(dst + ((size_t)i << 2), _buffer);
    }
}



void
decx::rcp::CPUK::_template_normalize_fp32_cpy2D(const float* src, 
                                                float* dst, 
                                                const size_t len, 
                                                const uint pitchsrc, 
                                                const uint pitchdst, 
                                                const uint width,
                                                const uint height)
{
    float sum = 0;
    decx::rcp::CPUK::_template_sum_vec8_fp32(src, len, &sum);
    size_t dex_src = 0, dex_dst = 0;
    const float _sub = sum / ((float)width * (float)height);
    float buffer;
    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < width; ++j) {
            dex_src = (size_t)i * pitchsrc + (size_t)j;
            dex_dst = (size_t)i * pitchdst + (size_t)j;
            buffer = src[dex_src];
            buffer -= _sub;
            dst[dex_dst] = buffer;
        }
    }
}



void
decx::rcp::CPUK::_template_normalize_uint8_cpy2D(const uint8_t* src, 
                                                float* dst, 
                                                const size_t len, 
                                                const uint pitchsrc, 
                                                const uint pitchdst, 
                                                const uint width,
                                                const uint height)
{
    int sum = 0;
    size_t dex_src = 0, dex_dst = 0;

    //decx::rcp::CPUK::_template_sum_vec4_fp64(src, len, &sum);
    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < width; ++j) {
            dex_src = (size_t)i * pitchsrc + (size_t)j;
            dex_dst = (size_t)i * pitchdst + (size_t)j;
            sum += src[dex_src];
        }
    }

    dex_src = 0;
    dex_dst = 0;
    const float _sub = (float)sum / ((float)width * (float)height);
    float buffer;
    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < width; ++j) {
            dex_src = (size_t)i * pitchsrc + (size_t)j;
            dex_dst = (size_t)i * pitchdst + (size_t)j;
            buffer = (float)(src[dex_src]);
            buffer -= _sub;
            dst[dex_dst] = buffer;
        }
    }
}



void 
decx::rcp::CPUK::_template_normalize_fp32_vec8_cpy2D(const float* src, 
                                                    float* dst, 
                                                    const size_t len, 
                                                    const uint pitchsrc, 
                                                    const uint pitchdst, 
                                                    const uint width, 
                                                    const uint height)
{
    float sum = 0;
    decx::rcp::CPUK::_template_sq_sum_vec8_fp32(src, len, &sum);

    __m256 _buffer, _sub = _mm256_set1_ps((float)width * (float)height / sum);
    size_t dex_src = 0, dex_dst = 0;
    uint _cpy_width = GetSmaller(pitchsrc, pitchdst);

    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < _cpy_width; ++j) {
            dex_src = ((size_t)i * pitchsrc + (size_t)j) << 3;
            dex_dst = ((size_t)i * pitchdst + (size_t)j) << 3;
            _buffer = _mm256_load_ps(src + dex_src);
            _buffer = _mm256_sub_ps(_buffer, _sub);
            _mm256_store_ps(dst + dex_dst, _buffer);
        }
    }
}




void 
decx::rcp::CPUK::_template_normalize_fp32_vec8_cpy2D_inplace(float* src, 
                                                    const size_t len, 
                                                    const uint width,
                                                    const uint pitch,
                                                    const uint height)
{
    float sum = 0;
    decx::rcp::CPUK::_template_sq_sum_vec8_fp32(src, len, &sum);

    __m256 _buffer, _sub = _mm256_set1_ps((float)width * (float)height / sum);
    size_t dex = 0;

    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < pitch; ++j) {
            _buffer = _mm256_load_ps(src + (dex << 3));
            _buffer = _mm256_sub_ps(_buffer, _sub);
            _mm256_store_ps(src + (dex << 3), _buffer);
            ++dex;
        }
    }
}
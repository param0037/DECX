/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "edge_det_ops.h"
#include "../../../DSP/regional/regional_comparision/CPU/rcp_sliding_window_avx_ops.h"


#define _22_5DEG_RAD 0.392699075
#define _67_5DEG_RAD 1.178097225
#define _112_5DEG_RAD 1.963495375
#define _157_5DEG_RAD 2.748893525

#define _45DEG_RAD 0.78539816
#define _90DEG_RAD 1.57079633
#define _135DEG_RAD 2.35619449


namespace decx {
    namespace vis {
        namespace CPUK {
            _THREAD_CALL_ inline __m256 normalize_direction(const decx::utils::simd::xmm256_reg recv_R)
            {
                decx::utils::simd::xmm256_reg reg0, res;

                reg0._vf = _mm256_cmp_ps(recv_R._vf, _mm256_set1_ps(0), _CMP_LT_OS);
                reg0._vi = _mm256_and_si256(reg0._vi, _mm256_castps_si256(_mm256_set1_ps(3.14159265359)));
                res._vf = _mm256_add_ps(_mm256_add_ps(recv_R._vf, reg0._vf), _mm256_set1_ps(_22_5DEG_RAD));
                reg0._vf = _mm256_cmp_ps(res._vf, _mm256_set1_ps(3.14159265359), _CMP_GE_OS);
                reg0._vi = _mm256_and_si256(reg0._vi, _mm256_castps_si256(_mm256_set1_ps(3.14159265359)));
                res._vf = _mm256_sub_ps(res._vf, reg0._vf);

                return res._vf;
            }
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::CPUK::_Edge_Detector_Post_processing(const float* __restrict      G_info_map, 
                                                const float* __restrict      dir_info_map, 
                                                float* __restrict            _cache,
                                                uint64_t*                    dst, 
                                                const uint                   WG, 
                                                const uint                   WD, 
                                                const uint                   Wdst,
                                                const uint2                  proc_dims, 
                                                const float2                 _thres)
{
    size_t dex_G = 0, dex_D = 0, dex_dst = 0;
    bool is_max = false, is_keep = false;

    decx::utils::simd::xmm256_reg recv_G, recv_R, reg, reg1;
    __m256i _vgather_addr_1s, _vgather_addr_pitch;
    const __m256i _shuffle_var = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i _v_addr = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 _cmp_lane = _mm256_setzero_ps(), _cmp_res, _cmp_res_tmp;
    uint64_t res;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_G = i * WG;
        dex_D = i * WD;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) 
        {
            recv_G._vf = _mm256_loadu_ps(G_info_map + dex_G + 1 + WG);
            recv_R._vf = _mm256_load_ps(dir_info_map + dex_D);

            _mm256_store_ps(_cache, _mm256_load_ps(G_info_map + dex_G));                    _mm256_store_ps(_cache + 8, _mm256_load_ps(G_info_map + dex_G + 8));
            _mm256_store_ps(_cache + 16, _mm256_load_ps(G_info_map + dex_G + WG));          _mm256_store_ps(_cache + 24, _mm256_load_ps(G_info_map + dex_G + WG + 8));
            _mm256_store_ps(_cache + 32, _mm256_load_ps(G_info_map + dex_G + WG * 2));      _mm256_store_ps(_cache + 40, _mm256_load_ps(G_info_map + dex_G + WG * 2 + 8));
            
            recv_R._vf = decx::vis::CPUK::normalize_direction(recv_R);

            reg._vf = _mm256_cmp_ps(recv_R._vf, _mm256_set1_ps(_135DEG_RAD), _CMP_GE_OS);
            _vgather_addr_1s = _mm256_or_si256(reg._vi, _mm256_set1_epi32(1));
            reg._vf = _mm256_cmp_ps(recv_R._vf, _mm256_set1_ps(_90DEG_RAD), _CMP_GE_OS);
            reg._vi = _mm256_and_si256(reg._vi, _mm256_castps_si256(_mm256_cmp_ps(recv_R._vf, _mm256_set1_ps(_135DEG_RAD), _CMP_LT_OS)));
            _vgather_addr_1s = _mm256_andnot_si256(reg._vi, _vgather_addr_1s);

            reg._vf = _mm256_cmp_ps(recv_R._vf, _mm256_set1_ps(_45DEG_RAD), _CMP_LT_OS);
            _vgather_addr_pitch = _mm256_andnot_si256(reg._vi, _mm256_set1_epi32(16));

            _vgather_addr_1s = _mm256_add_epi32(_vgather_addr_1s, _vgather_addr_pitch);
            _cmp_lane = _mm256_i32gather_ps(_cache + 17, _mm256_add_epi32(_vgather_addr_1s, _v_addr), 4);
            
            _cmp_res = _mm256_cmp_ps(recv_G._vf, _cmp_lane, _CMP_GE_OS);

            _cmp_lane = _mm256_i32gather_ps(_cache + 17, _mm256_sub_epi32(_v_addr, _vgather_addr_1s), 4);
            reg._vf = _mm256_cmp_ps(recv_G._vf, _cmp_lane, _CMP_GT_OS);
            
            _cmp_res = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(_cmp_res), reg._vi));

            reg._vf = _mm256_cmp_ps(recv_G._vf, _mm256_set1_ps(_thres.y), _CMP_GT_OS);
            _cmp_res = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(_cmp_res), reg._vi));

            // weak edges processing
            _cmp_res_tmp = _mm256_cmp_ps(_mm256_load_ps(_cache), _mm256_set1_ps(_thres.y), _CMP_GT_OS);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 1),  _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 2),  _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 16), _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 18), _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 24), _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 25), _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            _cmp_res_tmp = _mm256_or_ps(_mm256_cmp_ps(_mm256_loadu_ps(_cache + 26), _mm256_set1_ps(_thres.y), _CMP_GT_OS), _cmp_res_tmp);
            
            _cmp_res = _mm256_and_ps(_cmp_res, _cmp_res_tmp);
            _cmp_res = _mm256_and_ps(_cmp_res, _mm256_cmp_ps(recv_G._vf, _mm256_set1_ps(_thres.x), _CMP_GT_OS));

            reg._vi = _mm256_shuffle_epi8(_mm256_castps_si256(_cmp_res), _shuffle_var);
            reg._vi = _mm256_permutevar8x32_epi32(reg._vi, _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7));
            res = (_mm256_extract_epi64(reg._vi, 0) & 0xFFFFFFFFFFFFFFFFU);
            dst[dex_dst] = res;

            dex_G += 8;
            dex_D += 8;
            ++dex_dst;
        }
    }
}
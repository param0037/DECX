/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "fragment_arrangment.h"

_DECX_API_
bool decx::utils::frag_manager_gen(decx::utils::frag_manager* src, const size_t _tot, const size_t _frag_num)
{
    src->total = _tot;
    src->frag_num = _frag_num;
    src->is_left = _tot % _frag_num;
    if (src->is_left) {
        src->frag_len = _tot / _frag_num;
        src->frag_left_over = _tot - (_frag_num - 1) * src->frag_len;
        return false;
    }
    else {
        src->frag_len = _tot / _frag_num;
        src->frag_left_over = 0;
        return true;
    }
}


_DECX_API_
bool decx::utils::frag_manager_gen_from_fragLen(decx::utils::frag_manager* src, const size_t _tot, const size_t _frag_len)
{
    src->total = _tot;
    src->frag_len = _frag_len;
    if (_tot % _frag_len) {     // is left
        src->is_left = true;
        src->frag_num = _tot / _frag_len + 1;
        src->frag_left_over = _tot % _frag_len;
        return false;
    }
    else {
        src->is_left = false;
        src->frag_num = _tot / _frag_len;
        src->frag_left_over = _tot % _frag_len;
        return true;
    }
}


_DECX_API_
bool decx::utils::frag_manager_gen_Nx(decx::utils::frag_manager* src, const size_t _tot,
    const size_t _frag_num, const uint N)
{
    src->total = _tot;
    src->frag_num = _frag_num;
    if (_tot % N) {
        src->is_left = true;
        uint new_tot = _tot / N;
        src->frag_len = new_tot / _frag_num * N;
        src->frag_left_over = _tot - (_frag_num - 1) * src->frag_len;
        return false;
    }
    else {
        uint new_tot = _tot / N;
        if (new_tot % _frag_num) {
            src->is_left = true;
            src->frag_len = new_tot / _frag_num * N;
            src->frag_left_over = _tot - (_frag_num - 1) * src->frag_len;
            return false;
        }
        else {
            src->is_left = false;
            src->frag_len = _tot / _frag_num;
            src->frag_left_over = 0;
            return true;
        }
    }
}



// --------------------------- 2D --------------------------------------


float decx::_get_ratio_grater_than_one(const uint X, const uint Y)
{
    if (X > Y) {
        return ((float)X / (float)Y);
    }
    else {
        return ((float)Y / (float)X);
    }
}



uint decx::_get_mid_factors_pow2(const uint x)
{
    float expect_fac = sqrtf((float)x);
    uint pow = decx::utils::_GetHighest_abd(expect_fac);
    uint fac_R = 1 << pow;
    uint fac_L = 1 << (pow - 1);
    if (expect_fac - fac_L > fac_R - expect_fac) {
        return fac_R;
    }
    else {
        return fac_L;
    }
}



uint2 decx::find_closest_factors_by_ratio(const uint x, const uint2 pair)
{
    uint start_inc = (uint)sqrtf(x * (float)pair.y / (float)pair.x);
    uint start_dec = start_inc, closest_res;
    while (true) {
        if (x % start_inc) { ++start_inc; }
        else {
            closest_res = start_inc;
            break;
        }
        if (x % start_dec) { --start_dec; }
        else {
            closest_res = start_dec;
            break;
        }
    }
    return make_uint2(x / closest_res, closest_res);
}




uint decx::_fit_factor_to_no_reminder_inc(const uint __base, const uint __devided)
{
    uint base = __base + 1;
    while (base < __devided)
    {
        if (__devided % base == 0) {
            return base;
        }
        ++base;
    }
    return base;
}


uint2 decx::find_closest_factors_by_ratio_pow2(const uint x, const uint2 pair, bool pow2_on_x)
{
    const float mid_fac = sqrtf((float)x);
    const float ratio_GT_one = _get_ratio_grater_than_one(pair.x, pair.y);

    if (pow2_on_x) {
        // decide which one is greater
        float start_factor = pair.x < pair.y ? (mid_fac / ratio_GT_one) : (mid_fac * ratio_GT_one);

        float expect_fac = decx::utils::clamp_max<float>(decx::utils::clamp_min<float>(start_factor, 1), x);

        uint pow = decx::utils::_GetHighest(expect_fac);
        uint fac_R = (1 << pow);
        uint fac_L = (1 << (pow - 1));

        uint cloest_fac_pow2 = ((expect_fac - fac_L) > (fac_R - expect_fac)) ? fac_R : fac_L;
        make_uint2(cloest_fac_pow2, x / cloest_fac_pow2);
    }
    else {
        // decide which one is greater
        float start_factor = pair.x > pair.y ? (mid_fac / ratio_GT_one) : (mid_fac * ratio_GT_one);

        float expect_fac = decx::utils::clamp_max<float>(decx::utils::clamp_min<float>(start_factor, 1), x);

        uint pow = decx::utils::_GetHighest(expect_fac);
        uint fac_R = (1 << pow);
        uint fac_L = (1 << (pow - 1));

        uint cloest_fac_pow2 = ((expect_fac - fac_L) > (fac_R - expect_fac)) ? fac_R : fac_L;
        return make_uint2(x / cloest_fac_pow2, cloest_fac_pow2);
    }
}



void decx::utils::thread2D_arrangement_advisor_for_GEMM(uint2* thr_arrange, const uint total_thr_num, const uint2 proc_dims)
{
    // approximately a square, arrange the same number of threads noth on width and height if possible
    const float ratio = decx::_get_ratio_grater_than_one(proc_dims.x, proc_dims.y);
    if (ratio < 1.5) {
        uint fac1 = decx::_get_mid_factors_pow2(total_thr_num);
        thr_arrange->x = fac1;                      // width
        thr_arrange->y = total_thr_num / fac1;      // height
    }
    else {       // It is a rectangle
        *thr_arrange = decx::find_closest_factors_by_ratio_pow2(total_thr_num, proc_dims, true);
    }
}
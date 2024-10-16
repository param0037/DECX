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


#include "fragment_arrangment.h"


bool decx::utils::frag_manager_gen(decx::utils::frag_manager* src, const uint64_t _tot, const uint64_t _frag_num)
{
    src->total = _tot;

    if (_tot > _frag_num)
    {
        src->frag_num = _frag_num;
        src->is_left = _tot % _frag_num;
        bool res;
        if (src->is_left) {
            src->frag_len = _tot / _frag_num;
            src->frag_left_over = _tot - (_frag_num - 1) * src->frag_len;
            res = false;
        }
        else {
            src->frag_len = _tot / _frag_num;
            src->frag_left_over = 0;
            res = true;
        }
        src->last_frag_len = src->is_left ? src->frag_left_over : src->frag_len;
        return res;
    }
    else{
        src->frag_num = _tot;
        src->frag_len = 1;
        src->frag_left_over = 0;
        src->is_left = 0;
        src->last_frag_len = 1;
        return 0;
    }
}


bool decx::utils::frag_manager_gen_from_fragLen(decx::utils::frag_manager* src, const uint64_t _tot, const uint64_t _frag_len)
{
    // src->total = _tot;
    // src->frag_len = _frag_len;
    // bool res;
    // if (_tot % _frag_len) {     // is left
    //     src->is_left = true;
    //     src->frag_num = _tot / _frag_len + 1;
    //     src->frag_left_over = _tot % _frag_len;
    //     res = false;
    // }
    // else {
    //     src->is_left = false;
    //     src->frag_num = _tot / _frag_len;
    //     src->frag_left_over = _tot % _frag_len;
    //     res = true;
    // }
    // src->last_frag_len = src->is_left ? src->frag_left_over : src->frag_len;
    // return res;

    src->total = _tot;

    if (_tot > _frag_len)
    {
        src->frag_len = _frag_len;
        src->is_left = _tot % _frag_len;
        src->frag_num = _tot / _frag_len;
        src->last_frag_len = _tot - (src->frag_num - 1) * src->frag_len;
        return src->is_left;
    }
    else{
        src->frag_num = 1;
        src->frag_len = _tot;
        src->is_left = 0;
        src->last_frag_len = src->frag_len;
        return 0;
    }
}


bool decx::utils::frag_manager_gen_Nx(decx::utils::frag_manager* src, const uint64_t _tot,
    const uint64_t _frag_num, const uint32_t N)
{
    if (_tot < N){
        src->total = _tot;
        src->frag_num = 1;
        src->frag_len = _tot;
        src->frag_left_over = 0;
        src->last_frag_len = _tot;
        src->is_left = 0;
        return 0;
    }
    uint64_t aligned_tot = decx::utils::ceil<uint64_t>(_tot, N);
    decx::utils::frag_manager_gen(src, aligned_tot, _frag_num);

    src->frag_len *= N;
    src->total = _tot;
    src->frag_left_over = src->total - (src->frag_num - 1) * src->frag_len;
    src->last_frag_len = src->is_left ? src->frag_left_over : src->frag_len;

    return src->is_left;
}



// something wrong, for some inputs it fails to give the right results
void decx::utils::thread2D_arrangement_advisor(uint2*              thr_arrange, 
                                               const uint32_t      total_thr_num, 
                                               const uint2         proc_dims)
{
    if (total_thr_num > 1) {
        const float k = (float)proc_dims.x / (float)proc_dims.y;
        const uint32_t base = roundf(sqrtf((float)total_thr_num / k));

        if (base < 2) {
            *thr_arrange = make_uint2(total_thr_num, 1);
        }
        else if (base > total_thr_num) {
            *thr_arrange = make_uint2(1, total_thr_num);
        }
        else {
            uint32_t base_var_ni = base;
            uint32_t base_var_nz = base;
            // Towards infinity
            while (total_thr_num % base_var_ni) {
                ++base_var_ni;
            }
            // Towards zero
            while (total_thr_num % base_var_nz) {
                --base_var_nz;
            }
            // Get the differences and find the minimal
            uint32_t diff_ni = fabs(base - base_var_ni);
            uint32_t diff_nz = fabs(base - base_var_nz);
            uint32_t candidate_base = diff_ni > diff_nz ? base_var_nz : base_var_ni;
            thr_arrange->y = candidate_base;
            thr_arrange->x = total_thr_num / candidate_base;
        }
    }
    else {
        *thr_arrange = make_uint2(1, 1);
    }
}

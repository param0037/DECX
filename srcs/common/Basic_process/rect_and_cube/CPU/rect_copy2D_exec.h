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


#ifndef _RECT_COPY2D_EXEC_H_
#define _RECT_COPY2D_EXEC_H_

#include "../../../../classes/classes_util.h"
#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/utils/fragment_arrangment.h"


namespace decx
{
    /**
    * This function copy data from a matrix to another matrix(larger) to make zero padding
    * WARNING:
    * The size of destination matrix must be larger than that of copying area
    *       ------> y
    *    ___________________________
    * | |   A___________________    |
    * | |  |                    |   |
    * | |  |        data        |   |       A(start.x, start.y)
    * V |  |                    |   |       B(end.x, end.y)
    * x |  |___________________B|   |
    *   |                           |
    *   |___________________________|
    */
    template <typename T>
    static _THREAD_GENERAL_ void _general_copy2D_BC(const T* __restrict src, T* __restrict dst, const uint2 start, const uint2 end,
        const uint32_t Wsrc, const uint32_t Wdst);


    /*
    * @param cpy_area : .x -> the width of copy area, in __m256 (vector of 256 bits); .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    template <typename T> _THREAD_GENERAL_
    static void _cpy2D_plane(const T* __restrict src, T* dst, const uint32_t Wsrc, const uint32_t Wdst, const uint2 cpy_area);


    /*
    * @param cpy_area : .x -> the width of copy area, in own element; .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    template <typename _T_ele>
    static void _cpy2D_anybit_caller(_T_ele* src, _T_ele* dst, const uint32_t Wsrc, const uint32_t Wdst, const uint2 cpy_area);
}


/*
* @param Wsrc : width of source matrix, in its own element
* @param Wdst : width of destinated matrix, in its own element
*/
template <typename T>
_THREAD_GENERAL_ void decx::_general_copy2D_BC(const T* __restrict           src, 
                                               T* __restrict           dst, 
                                               const uint2             start, 
                                               const uint2             end, 
                                               const uint32_t              Wsrc, 
                                               const uint32_t              Wdst)
{
    // ~.x : width; ~.y : height
    uint2 copy_dim = make_uint2(end.y - start.y, end.x - start.x);

    T* start_dst = DECX_PTR_SHF_XY<T, T>(dst, start, Wdst);
    
    for (int i = 0; i < copy_dim.y; ++i) {
        memcpy((void*)(start_dst + i * Wdst), (void*)(src + i * Wsrc), copy_dim.x * sizeof(T));
    }
}


template <typename T> _THREAD_GENERAL_
void decx::_cpy2D_plane(const T* __restrict src, T* __restrict dst, const uint32_t Wsrc, const uint32_t Wdst, const uint2 cpy_area)
{
    uint64_t  r_dex_src = 0, r_dex_dst = 0;

    for (int i = 0; i < cpy_area.y; ++i) {
        memcpy(dst + r_dex_dst, src + r_dex_src, cpy_area.x * sizeof(T));
        r_dex_src += Wsrc;          r_dex_dst += Wdst;
    }
}



template <typename _T_ele>
static void decx::_cpy2D_anybit_caller(_T_ele* src, _T_ele* dst, const uint32_t Wsrc, const uint32_t Wdst, const uint2 cpy_area)
{
    const uint32_t conc_thr = (uint32_t)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, cpy_area.y, conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    
    _T_ele* tmp_src = src, * tmp_dst = dst;
    for (int i = 0; i < conc_thr - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::_cpy2D_plane<_T_ele>,
            tmp_src, tmp_dst, Wsrc, Wdst, make_uint2(cpy_area.x, f_mgr.frag_len));

        tmp_src += (uint64_t)Wsrc * (uint64_t)f_mgr.frag_len;
        tmp_dst += (uint64_t)Wdst * (uint64_t)f_mgr.frag_len;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(decx::_cpy2D_plane<_T_ele>,
        tmp_src, tmp_dst, Wsrc, Wdst, make_uint2(cpy_area.x, _L));

    for (int i = 0; i < conc_thr; ++i) {
        t1D._async_thread[i].get();
    }
}



namespace decx
{
    _THREAD_FUNCTION_
    /*
    * @param cpy_area : .x -> the width of copy area, in __m256 (vector of 256 bits); .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    void _cpy2D_plane_array(const float* src, float* dst, const uint32_t Wsrc, const uint32_t Wdst, const uint2 cpy_area);


    /*
    * @param cpy_area : .x -> the width of copy area, in own element; .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    template <typename _T_ele>
    static void _cpy2D_PA_32bit_caller(_T_ele* src, _T_ele* dst, const uint32_t Wsrc, const uint32_t Wdst, const uint2 cpy_area);
}



#endif
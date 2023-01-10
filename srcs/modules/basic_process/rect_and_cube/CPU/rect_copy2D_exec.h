/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _RECT_COPY2D_EXEC_H_
#define _RECT_COPY2D_EXEC_H_

#include "../../../classes/classes_util.h"
#include "../../../core/basic.h"
#include "../../../core/utils/leftovers.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"


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
        const uint Wsrc, const uint Wdst);


    /*
    * @param cpy_area : .x -> the width of copy area, in __m256 (vector of 256 bits); .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    template <typename T> _THREAD_GENERAL_
    static void _cpy2D_plane(const T* __restrict src, T* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_area);


    /*
    * @param cpy_area : .x -> the width of copy area, in own element; .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    template <typename _T_ele>
    static void _cpy2D_32bit_caller(_T_ele* src, _T_ele* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_area);
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
                                               const uint              Wsrc, 
                                               const uint              Wdst)
{
    // ~.x : width; ~.y : height
    uint2 copy_dim = make_uint2(end.y - start.y, end.x - start.x);

    T* start_dst = DECX_PTR_SHF_XY<T, T>(dst, start, Wdst);
    
    for (int i = 0; i < copy_dim.y; ++i) {
        memcpy((void*)(start_dst + i * Wdst), (void*)(src + i * Wsrc), copy_dim.x * sizeof(T));
    }
}


template <typename T> _THREAD_GENERAL_
void decx::_cpy2D_plane(const T* __restrict src, T* __restrict dst, const uint Wsrc, const uint Wdst, const uint2 cpy_area)
{
    size_t r_dex_src = 0, r_dex_dst = 0;

    for (int i = 0; i < cpy_area.y; ++i) {
        memcpy(dst + r_dex_dst, src + r_dex_src, cpy_area.x * sizeof(T));
        r_dex_src += Wsrc;          r_dex_dst += Wdst;
    }
}



template <typename _T_ele>
static void decx::_cpy2D_32bit_caller(_T_ele* src, _T_ele* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_area)
{
    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, cpy_area.y, conc_thr);

    std::future<void>* fut = new std::future<void>[conc_thr];
    
    if (f_mgr.frag_left_over != 0) {
        _T_ele* tmp_src = src, * tmp_dst = dst;
        for (int i = 0; i < conc_thr - 1; ++i) {
            fut[i] = decx::cpu::register_task(&decx::thread_pool,decx::_cpy2D_plane<_T_ele>,
                tmp_src, tmp_dst, Wsrc, Wdst, make_uint2(cpy_area.x, f_mgr.frag_len));
            
            tmp_src += (size_t)Wsrc * (size_t)f_mgr.frag_len;
            tmp_dst += (size_t)Wdst * (size_t)f_mgr.frag_len;
        }
        fut[conc_thr - 1] = decx::cpu::register_task(&decx::thread_pool,decx::_cpy2D_plane<_T_ele>,
            tmp_src, tmp_dst, Wsrc, Wdst, make_uint2(cpy_area.x, f_mgr.frag_left_over));
    }
    else {
        _T_ele* tmp_src = src, * tmp_dst = dst;
        for (int i = 0; i < conc_thr; ++i) {
            fut[i] = decx::cpu::register_task(&decx::thread_pool,decx::_cpy2D_plane<_T_ele>,
                tmp_src, tmp_dst, Wsrc, Wdst, make_uint2(cpy_area.x, f_mgr.frag_len));
            
            tmp_src += (size_t)Wsrc * (size_t)f_mgr.frag_len;
            tmp_dst += (size_t)Wdst * (size_t)f_mgr.frag_len;
        }
    }

    for (int i = 0; i < conc_thr; ++i) {
        fut[i].get();
    }
    delete[] fut;
}



namespace decx
{
    _THREAD_FUNCTION_
    /*
    * @param cpy_area : .x -> the width of copy area, in __m256 (vector of 256 bits); .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    void _cpy2D_plane_array(const float* src, float* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_area);


    /*
    * @param cpy_area : .x -> the width of copy area, in own element; .y -> the height of copy area
    * @param Wsrc : pitch of src, in its own element
    * @param Wdst : pitch of dst, in its own element
    */
    template <typename _T_ele>
    static void _cpy2D_PA_32bit_caller(_T_ele* src, _T_ele* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_area);
}



#endif
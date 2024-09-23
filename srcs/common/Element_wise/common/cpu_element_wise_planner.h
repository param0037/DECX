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

#ifndef _CPU_ELEMENT_WISE_PLANNER_H_
#define _CPU_ELEMENT_WISE_PLANNER_H_

#include "../../basic.h"
#include "../../FMGR/fragment_arrangment.h"
#include "../../../modules/core/configs/config.h"
#include "../../../modules/core/thread_management/thread_arrange.h"
#include "../../../modules/core/thread_management/thread_pool.h"
#include "element_wise_base.h"



namespace decx
{
    class cpu_ElementWise1D_planner;
    class cpu_ElementWise2D_planner;
}


class decx::cpu_ElementWise1D_planner : public decx::element_wise_base_1D
{
protected:
    uint32_t _concurrency;

    uint64_t _min_thread_proc;

    decx::utils::frag_manager _fmgr;

    bool changed(const uint32_t conc, const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc) const;

public:
    cpu_ElementWise1D_planner() {}


    void plan(const uint32_t conc, const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_);

    
    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_unary(FuncType&& f, const _type_in* src, _type_out* dst, decx::utils::_thr_1D* t1D, Args&& ...additional)
    {
        const _type_in* loc_src = src;
        _type_out* loc_dst = dst;

        for (int32_t i = 0; i < this->_fmgr.frag_num; ++i){
            const uint64_t _proc_len_v = i < this->_fmgr.frag_num - 1 ? this->_fmgr.frag_len : this->_fmgr.last_frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default(f, loc_src, loc_dst, _proc_len_v, additional...);

            loc_src += _proc_len_v * this->_alignment;
            loc_dst += _proc_len_v * this->_alignment;
        }

        t1D->__sync_all_threads(make_uint2(0, this->_fmgr.frag_num));
    }


    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_binary(FuncType&& f, const _type_in* src1, const _type_in* src2, _type_out* dst, decx::utils::_thr_1D* t1D, Args&& ...additional)
    {
        uint64_t dex_src = 0, dex_dst = 0;

        for (int32_t i = 0; i < this->_fmgr.frag_num; ++i){
            const uint64_t _proc_len_v = i < this->_fmgr.frag_num - 1 ? this->_fmgr.frag_len : this->_fmgr.last_frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default(f, src1 + dex_src, src2 + dex_src, dst + dex_dst, _proc_len_v, additional...);

            dex_src += _proc_len_v * this->_alignment;
            dex_dst += _proc_len_v * this->_alignment;
        }

        t1D->__sync_all_threads(make_uint2(0, this->_fmgr.frag_num));
    }
};


class decx::cpu_ElementWise2D_planner : public decx::element_wise_base_2D
{
protected:
    uint32_t _concurrency;

    uint64_t _min_thread_proc;

    decx::utils::frag_manager _fmgr_WH[2];

    bool changed(const uint32_t conc, const uint2 proc_dims, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc) const;

public:
    cpu_ElementWise2D_planner() {}


    void plan(const uint32_t conc, const uint2 proc_dims, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_);

    uint2 _thread_dist;

    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_unary(FuncType&& f, const _type_in* src, _type_out* dst, const uint32_t Wsrc, const uint32_t Wdst, 
        decx::utils::_thr_1D* t1D, Args&& ...additional)
    {
        const _type_in* loc_src = src;
        _type_out* loc_dst = dst;

        uint32_t _thr_cnt = 0;

        for (int32_t i = 0; i < this->_thread_dist.y; ++i){
            loc_src = src + Wsrc * i * this->_fmgr_WH[1].frag_len;
            loc_dst = dst + Wdst * i * this->_fmgr_WH[1].frag_len;
            for (int32_t j = 0; j < this->_thread_dist.x; ++j){
                uint2 proc_dims_v = 
                    make_uint2(j < this->_thread_dist.x - 1 ? this->_fmgr_WH[0].frag_len : this->_fmgr_WH[0].last_frag_len,
                            i < this->_thread_dist.y - 1 ? this->_fmgr_WH[1].frag_len : this->_fmgr_WH[1].last_frag_len);

                t1D->_async_thread[_thr_cnt] = decx::cpu::register_task_default(f, loc_src, loc_dst, proc_dims_v, Wsrc, Wdst, additional...);
                
                loc_src += this->_fmgr_WH[0].frag_len * this->_alignment;
                loc_dst += this->_fmgr_WH[0].frag_len * this->_alignment;
                ++_thr_cnt;
            }
        }
        t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
    }


    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_binary(FuncType&& f, const _type_in* src1, const _type_in* src2, _type_out* dst, const uint32_t Wsrc, const uint32_t Wdst, 
        decx::utils::_thr_1D* t1D, Args&& ...additional)
    {
        uint64_t dex_src = 0, dex_dst = 0;
        uint32_t _thr_cnt = 0;

        for (int32_t i = 0; i < this->_thread_dist.y; ++i)
        {
            dex_src = Wsrc * i * this->_fmgr_WH[1].frag_len;
            dex_dst = Wdst * i * this->_fmgr_WH[1].frag_len;

            for (int32_t j = 0; j < this->_thread_dist.x; ++j){
                uint2 proc_dims_v = 
                    make_uint2(j < this->_thread_dist.x - 1 ? this->_fmgr_WH[0].frag_len : this->_fmgr_WH[0].last_frag_len,
                            i < this->_thread_dist.y - 1 ? this->_fmgr_WH[1].frag_len : this->_fmgr_WH[1].last_frag_len);

                t1D->_async_thread[_thr_cnt] = decx::cpu::register_task_default(f, src1 + dex_src, src2 + dex_src, dst + dex_dst, proc_dims_v, 
                                    Wsrc, Wdst, additional...);
                
                dex_src += this->_fmgr_WH[0].frag_len * this->_alignment;
                dex_dst += this->_fmgr_WH[0].frag_len * this->_alignment;
                ++_thr_cnt;
            }
        }
        t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
    }
};

#endif
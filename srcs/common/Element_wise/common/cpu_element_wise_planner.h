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

#include <basic.h>
#include <FMGR/fragment_arrangment.h>
#include <configs/config.h>
#include <thread_management/thread_arrange.h>
#include <thread_management/thread_pool.h>
#include "element_wise_base.h"
#include <Concurrent/thread_argument.h>

#include <PtrInfo.h>
#include <Concurrent/compute_loads_mgr.h>

namespace decx
{
    class cpu_ElementWise1D_planner;
    class cpu_ElementWise2D_planner;
}


#define EW_SLOT_ID_UNUSED            decx::TArg_var<int32_t>([&](const int32_t i){return 0;})
#define EW_SLOT_ID_MONOTONIC(offset) decx::TArg_var<int32_t>([&](const int32_t i){return i + (offset);})
#define EW_SLOT_ID_CUSTOM(func)      decx::TArg_var<int32_t>((func))


class decx::cpu_ElementWise1D_planner : public decx::element_wise_base_1D
{
protected:
    uint32_t _concurrency;

    uint64_t _min_thread_proc;

    decx::utils::frag_manager _fmgr;

    bool changed(const uint32_t conc, const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc) const;

    decx::utils::ComputeLoadsMgr _tasks;

public:
    cpu_ElementWise1D_planner() {}


    void plan(const uint32_t simd_align_byte, const uint32_t conc, const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_, const uint32_t extra_alignment = 1);


    uint64_t get_proc_len_by_id(const int32_t thread_id) const{
        return this->_fmgr.GetFragLenById(thread_id);
    }


    uint64_t get_proc_len_v_by_id(const int32_t thread_id) const{
        return decx::utils::idiv_ceil<uint64_t>(this->_fmgr.GetFragLenById(thread_id), this->_alignment);
    }


    const decx::utils::frag_manager* get_fmgr() const {return &this->_fmgr;}


    const uint8_t get_alignment() const {return this->_alignment;}


    template <typename FuncType, typename LambdaFunc_T, typename... Args> inline int32_t 
    Caller(FuncType&&                                      f, 
           const decx::cpu::ThreadDispatchMethod_e         method, 
           decx::ThreadArg_var<int32_t, LambdaFunc_T>&&    slot_id, 
           Args&&...                                       args)
    {
        int32_t rval = 0;

        for (int32_t i = 0; i < this->_fmgr.GetFragNum(); ++i){
            rval |= this->_tasks.AppendTask(std::forward<FuncType>(f), std::forward<Args>(args).value(i)...);
            this->_tasks.Back()->_slot_id = slot_id.value(i);
        }
        
        rval |= this->_tasks.RunAll();
        
        rval |= this->_tasks.SynchronizeAll();

        rval |= this->_tasks.ClearAll();

        return rval;
    }


    template <typename FuncType, typename LambdaFunc_T, typename... Args> static int32_t 
    sCaller(FuncType&&                                      f, 
            const decx::utils::frag_manager*                fmgr, 
            decx::utils::ComputeLoadsMgr*                   p_tasks_mgr, 
            const decx::cpu::ThreadDispatchMethod_e         method, 
            decx::ThreadArg_var<int32_t, LambdaFunc_T>&&    slot_id, 
            Args&&                                          ...args)
    {
        int32_t rval = 0;

        for (int32_t i = 0; i < fmgr->GetFragNum(); ++i){
            rval |= p_tasks_mgr->AppendTask(std::forward<FuncType>(f), std::forward<Args>(args).value(i)...);
            p_tasks_mgr->Back()->_slot_id = slot_id.value(i);
        }
        
        rval |= p_tasks_mgr->RunAll();
        
        rval |= p_tasks_mgr->SynchronizeAll();

        rval |= p_tasks_mgr->ClearAll();

        return rval;
    }


    ~cpu_ElementWise1D_planner();
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


    void plan(const uint32_t simd_align_byte, const uint32_t conc, const uint2 proc_dims, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_, const uint2 extra_alignment = make_uint2(1, 1));

    uint2 _thread_dist;


    uint2 get_proc_dims_by_id(const int32_t i, const int32_t j) const
    {
        return make_uint2(this->_fmgr_WH[0].GetFragLenById(j),
                          this->_fmgr_WH[1].GetFragLenById(i));
    }


    uint2 get_proc_dims_v_by_id(const int32_t i, const int32_t j) const
    {
        return make_uint2(decx::utils::idiv_ceil<uint32_t>(this->_fmgr_WH[0].GetFragLenById(j), this->_alignment),
                          this->_fmgr_WH[1].GetFragLenById(i));
    }


    template <typename FuncType, typename LambdaFunc_T, typename ...Args>
    inline void caller(FuncType&& f, 
                       decx::utils::Thr1D* t1D, 
                       const decx::cpu::ThreadDispatchMethod_e method,
                       decx::ThreadArg_var<int32_t, LambdaFunc_T>&& slot_id,
                       Args&& ...args)
    {
        uint32_t _thr_cnt = 0;

        for (int32_t i = 0; i < this->_thread_dist.y; ++i){
            for (int32_t j = 0; j < this->_thread_dist.x; ++j){
                t1D->_async_thread[_thr_cnt] = decx::cpu::RegisterTask(f, method, slot_id.value(i), args.value(i, j)...);
                ++_thr_cnt;
            }
        }
        t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
    }


    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_unary(FuncType&& f, const _type_in* src, _type_out* dst, const uint32_t Wsrc, const uint32_t Wdst, 
        decx::utils::Thr1D* t1D, Args&& ...additional)
    {
        const _type_in* loc_src = src;
        _type_out* loc_dst = dst;

        uint32_t _thr_cnt = 0;

        for (int32_t i = 0; i < this->_thread_dist.y; ++i){
            loc_src = src + Wsrc * i * this->_fmgr_WH[1].frag_len;
            loc_dst = dst + Wdst * i * this->_fmgr_WH[1].frag_len;
            for (int32_t j = 0; j < this->_thread_dist.x; ++j)
            {
                uint2 proc_dims = this->get_proc_dims_v_by_id(i, j);
                t1D->_async_thread[_thr_cnt] = decx::cpu::RegisterTaskLoadBalanced(f, loc_src, loc_dst, proc_dims, Wsrc, Wdst, additional...);
                
                loc_src += this->_fmgr_WH[0].frag_len;
                loc_dst += this->_fmgr_WH[0].frag_len;
                ++_thr_cnt;
            }
        }
        t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
    }


    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_binary(FuncType&& f, const _type_in* src1, const _type_in* src2, _type_out* dst, const uint32_t Wsrc, const uint32_t Wdst, 
        decx::utils::Thr1D* t1D, Args&& ...additional)
    {
        uint64_t dex_src = 0, dex_dst = 0;
        uint32_t _thr_cnt = 0;

        for (int32_t i = 0; i < this->_thread_dist.y; ++i)
        {
            dex_src = Wsrc * i * this->_fmgr_WH[1].frag_len;
            dex_dst = Wdst * i * this->_fmgr_WH[1].frag_len;

            for (int32_t j = 0; j < this->_thread_dist.x; ++j){
                uint2 proc_dims = 
                    make_uint2(j < this->_thread_dist.x - 1 ? this->_fmgr_WH[0].frag_len : this->_fmgr_WH[0].last_frag_len,
                            i < this->_thread_dist.y - 1 ? this->_fmgr_WH[1].frag_len : this->_fmgr_WH[1].last_frag_len);

                t1D->_async_thread[_thr_cnt] = decx::cpu::RegisterTaskLoadBalanced(f, src1 + dex_src, src2 + dex_src, dst + dex_dst, proc_dims, 
                                    Wsrc, Wdst, additional...);
                
                dex_src += this->_fmgr_WH[0].frag_len;
                dex_dst += this->_fmgr_WH[0].frag_len;
                ++_thr_cnt;
            }
        }
        t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
    }

};


#endif

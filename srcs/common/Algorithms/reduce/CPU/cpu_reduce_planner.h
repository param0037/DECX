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

#ifndef _CPU_REDUCE_PLANNER_H_
#define _CPU_REDUCE_PLANNER_H_

#include <basic.h>
#include <thread_management/thread_pool.h>
#include <Element_wise/common/cpu_element_wise_planner.h>
#include <allocators.h>
#include <SIMD/intrinsics_ops.h>
#include <thread_management/thread_arrange.h>

namespace decx
{
namespace reduce{
    class cpu_Reduce1D_Planner;
}
}


class decx::reduce::cpu_Reduce1D_Planner : public decx::cpu_ElementWise1D_planner
{
private:
    decx::PtrInfo<void> _shared_memory;

    decx::utils::simd::xmm256_reg _mask;

    void mask_gen_128b(const uint32_t l);
    void mask_gen_256b(const uint32_t l);

public:
    cpu_Reduce1D_Planner() {}


    const decx::utils::frag_manager* get_distribution() const{
        return &this->_fmgr;
    }


    _CRSR_ void alloc_shared_mem(const uint64_t size, de::DH* handle);


    void plan(const uint32_t conc, const uint64_t total, const uint8_t type_size_in, const uint8_t type_size_out,
        const uint64_t min_thread_proc=_EW_MIN_THREAD_PROC_DEFAULT_CPU_);


    // template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    // void caller_VVOO(FuncType&& f, const _type_in* p_in1, const _type_in* p_in2, _type_out* p_out1, 
    //     _type_out* p_out2, decx::utils::_thr_1D* t1D, Args&& ...additional);


    template <typename _ptr_type>
    _ptr_type* get_shared_mem() {
        return (_ptr_type*)this->_shared_memory.ptr;
    }
};


// template <typename FuncType,  typename _type_in, 
//           typename _type_out, class ...Args>
// inline void decx::reduce::cpu_Reduce1D_Planner::
// caller_VVOO(FuncType&&             f, 
//             const _type_in*        p_in1, 
//             const _type_in*        p_in2, 
//             _type_out*             p_out1, 
//             _type_out*             p_out2, 
//             decx::utils::_thr_1D*  t1D,
//             Args&&                 ...static_vars)
// {
//     const _type_in* loc_p_in1 = p_in1, *loc_p_in2 = p_in2;
//     _type_out* loc_p_out1 = p_out1, *loc_p_out2 = p_out2;

//     for (int32_t i = 0; i < this->_fmgr.frag_num; ++i){
//         const uint64_t _proc_len_v1 = this->_fmgr.get_frag_len_by_id(i);
//         t1D->_async_thread[i] = decx::cpu::register_task_default(
//             f, loc_p_in1, loc_p_in2, loc_p_out1, loc_p_out2, _proc_len_v1, static_vars...);
//         // printf("%llu, %llu, %llu, %llu, %llu\n", loc_p_in1, loc_p_in2, loc_p_out1, loc_p_out2, _proc_len_v1);
//         loc_p_in1 += _proc_len_v1;
//         loc_p_in2 += _proc_len_v1;
//         ++loc_p_out1;
//         ++loc_p_out2;
//     }
//     t1D->__sync_all_threads(make_uint2(0, this->_fmgr.frag_num));
// }


// template <typename FuncType, typename... Args> inline void 
// decx::reduce::cpu_Reduce1D_Planner::caller(FuncType&& f, 
//                                            decx::utils::_thr_1D* t1D, 
//                                            Args&&... args)
// {
//     for (int32_t i = 0; i < this->_fmgr.get_frag_num(); ++i){
//         t1D->_async_thread[i] = decx::cpu::register_task_default(f, args.value(i)...);
//     }
//     t1D->__sync_all_threads(make_uint2(0, this->_fmgr.frag_num));
// }


#endif

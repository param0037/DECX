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

#include "cpu_element_wise_planner.h"
#include <log_console.h>
#define MODULE_TAG "EW2D"

bool
decx::cpu_ElementWise1D_planner::changed(const uint32_t conc,
                                      const uint64_t total, 
                                      const uint8_t type_in_size, 
                                      const uint8_t type_out_size,
                                      const uint64_t min_thread_proc) const
{
    uint32_t cmp_conc = conc ^ this->_concurrency;
    uint64_t cmp_total = total ^ this->_total;
    uint32_t cmp_type_size = (((uint32_t)type_in_size << 8) | this->_type_in_size) ^ 
                             (((uint32_t)type_out_size << 8) | this->_type_out_size);
    uint32_t cmp_MTP = min_thread_proc ^ this->_min_thread_proc;

    return cmp_conc | cmp_total | (uint32_t)cmp_type_size | cmp_MTP;
}


void 
decx::cpu_ElementWise1D_planner::plan(const uint32_t simd_align_byte,
                                      const uint32_t conc,
                                      const uint64_t total, 
                                      const uint8_t type_in_size, 
                                      const uint8_t type_out_size,
                                      const uint64_t min_thread_proc,
                                      const uint32_t extra_alignment)
{
    if (this->changed(conc, total, type_in_size, type_out_size, min_thread_proc))
    {
        this->_type_in_size = type_in_size;
        this->_type_out_size = type_out_size;

        this->plan_alignment(simd_align_byte);      // Calculate this->_alignment

        this->_min_thread_proc = min_thread_proc;
        this->_total = total;
        if (extra_alignment > this->_alignment) {
            if (extra_alignment % this->_alignment != 0){
                DECX_LOG_ERR("Extra alignment is not divisible by the simd alignment");
                return;
            }
        }
        const uint32_t align_x = max(this->_alignment, extra_alignment);

        this->_total_v = decx::utils::idiv_ceil<uint64_t>(this->_total, this->_alignment);

        this->_concurrency = conc;

        if (this->_total / this->_concurrency > this->_min_thread_proc){
            decx::utils::frag_manager_gen_Nx(&this->_fmgr, this->_total, this->_concurrency, align_x);
        }
        else{
            const uint32_t real_conc = decx::utils::idiv_ceil<uint64_t>(this->_total, this->_min_thread_proc);
            decx::utils::frag_manager_gen_Nx(&this->_fmgr, this->_total, real_conc, align_x);
        }
    }
}


bool
decx::cpu_ElementWise2D_planner::changed(const uint32_t conc,
                                      const uint2 proc_dims, 
                                      const uint8_t type_in_size, 
                                      const uint8_t type_out_size,
                                      const uint64_t min_thread_proc) const
{
    uint32_t cmp_conc = conc ^ this->_concurrency;
    uint64_t cmp_total = proc_dims.x ^ this->_proc_dims.x;
    cmp_total |= proc_dims.y ^ this->_proc_dims.y;

    uint32_t cmp_type_size = (((uint32_t)type_in_size << 8) | this->_type_in_size) ^ 
                             (((uint32_t)type_out_size << 8) | this->_type_out_size);
    uint32_t cmp_MTP = min_thread_proc ^ this->_min_thread_proc;

    return cmp_conc | cmp_total | (uint32_t)cmp_type_size | cmp_MTP;
}


void decx::
cpu_ElementWise2D_planner::plan(const uint32_t  simd_align_byte,
                                const uint32_t  conc, 
                                const uint2     proc_dims, 
                                const uint8_t   type_in_size, 
                                const uint8_t   type_out_size,
                                const uint64_t  min_thread_proc,
                                const uint2     extra_alignment_WH)
{
    if (this->changed(conc, proc_dims, type_in_size, type_out_size, min_thread_proc))
    {
        this->_type_in_size = type_in_size;
        this->_type_out_size = type_out_size;

        this->_concurrency = conc;

        this->plan_alignment(simd_align_byte);      // Calculate this->_alignment

        this->_proc_dims = proc_dims;
        this->_min_thread_proc = min_thread_proc;

        if (extra_alignment_WH.x > this->_alignment) {
            if (extra_alignment_WH.x % this->_alignment != 0){
                DECX_LOG_ERR("Extra alignment is not divisible by the simd alignment");
                return;
            }
        }
        const uint32_t align_x = max(this->_alignment, extra_alignment_WH.x);

        this->_proc_w_v = decx::utils::idiv_ceil<uint32_t>(this->_proc_dims.x, this->_alignment);

        const uint64_t _total = static_cast<uint64_t>(this->_proc_dims.x) * static_cast<uint64_t>(this->_proc_dims.y);

        if ((_total / this->_concurrency) < this->_min_thread_proc){
            const uint32_t real_conc = decx::utils::idiv_ceil<uint64_t>(_total, this->_min_thread_proc);

            this->_thread_dist = make_uint2(1, real_conc);
            decx::utils::frag_manager_gen(this->_fmgr_WH, this->_proc_w_v, 1);
            decx::utils::frag_manager_gen_Nx(this->_fmgr_WH + 1, this->_proc_dims.y, real_conc, extra_alignment_WH.y);
        }
        else{
            if (this->_proc_dims.y < this->_concurrency){
                decx::utils::thread2D_arrangement_advisor(&this->_thread_dist, this->_concurrency, this->_proc_dims);
                decx::utils::frag_manager_gen_Nx(this->_fmgr_WH, this->_proc_w_v, this->_thread_dist.x, align_x / this->_alignment);
                decx::utils::frag_manager_gen_Nx(this->_fmgr_WH + 1, this->_proc_dims.y, this->_thread_dist.y, extra_alignment_WH.y);
            }
            else{
                this->_thread_dist = make_uint2(1, this->_concurrency);
                decx::utils::frag_manager_gen(this->_fmgr_WH, this->_proc_w_v, 1);
                decx::utils::frag_manager_gen_Nx(this->_fmgr_WH + 1, this->_proc_dims.y, this->_concurrency, extra_alignment_WH.y);
            }
        }
    }
}

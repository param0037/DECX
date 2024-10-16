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


#include "cpu_reduce_planner.h"


void decx::reduce::cpu_Reduce1D_Planner::mask_gen_256b(const uint32_t l)
{
    uint32_t i;
    switch (this->_type_in_size)
    {
    case 4:
        for (i = 0; i < 8 - l; ++i){
            this->_mask._arrui[i] = 0xffffffffU;
        } break;
    case 8:
        for (i = 0; i < 4 - l; ++i){
            this->_mask._arrull[i] = 0xffffffffffffffffU;
        } break;
    case 1:
        for (i = 0; i < 32 - l; ++i){
            this->_mask._arruc[i] = 255;
        } break;
    default: break;
    }
}


void decx::reduce::cpu_Reduce1D_Planner::mask_gen_128b(const uint32_t l)
{
    uint32_t i;
    switch (this->_type_in_size)
    {
    case 4:
        for (i = 0; i < 4 - l; ++i){
            this->_mask._vmm128[0]._arrui[i] = 0xffffffffU;
        } break;
    case 8:
        for (i = 0; i < 2 - l; ++i){
            this->_mask._vmm128[0]._arrull[i] = 0xffffffffffffffffU;
        } break;
    case 1:
        for (i = 0; i < 16 - l; ++i){
            this->_mask._vmm128[0]._arruc[i] = 255;
        } break;
    default: break;
    }
}


void decx::reduce::cpu_Reduce1D_Planner::
plan(const uint32_t conc,           const uint64_t total, 
     const uint8_t type_size_in,    const uint8_t type_size_out,
     const uint64_t min_thread_proc)
{
    decx::cpu_ElementWise1D_planner::plan(conc, total, type_size_in, type_size_out, min_thread_proc);

    const uint32_t _L = this->_total_v * this->_alignment - this->_total;
    
#if defined(__x86_64__) || defined(__i386__)
        this->mask_gen_256b(_L);
#endif
#if defined(__aarch64__) || defined(__arm__)
        this->mask_gen_128b(_L);
#endif
}


void decx::reduce::cpu_Reduce1D_Planner::
alloc_shared_mem(const uint64_t size, de::DH* handle)
{
    if (decx::alloc::_host_virtual_page_malloc(&this->_shared_memory, size)){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
}

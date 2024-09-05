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


#if defined(__x86_64__) || defined(__i386__)
#define _EW_MIN_THREAD_PROC_DEFAULT_CPU_ 256
#endif
#if defined(__aarch64__) || defined(__arm__)
#define _EW_MIN_THREAD_PROC_DEFAULT_CPU_ 128
#endif


#ifdef _DECX_CPU_PARTS_
namespace decx
{
    class cpu_ElementWise1D_planner;
    class cpu_ElementWise2D_planner;
}


class decx::cpu_ElementWise1D_planner : public decx::element_wise_base_1D
{
private:
    uint32_t _concurrency;

    uint64_t _min_thread_proc;

public:
    cpu_ElementWise1D_planner() {}


    void plan(const uint32_t conc, const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_);
};



class decx::cpu_ElementWise2D_planner : public decx::element_wise_base_2D
{
private:
    uint32_t _concurrency;

    uint64_t _min_thread_proc;

public:
    cpu_ElementWise2D_planner() {}


    void plan(const uint32_t conc, const uint2 proc_dims, const uint8_t _type_in_size, const uint8_t _type_out_size,
        const uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_);
};



#endif  // #ifdef _DECX_CPU_PARTS_


#endif
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

#ifndef _CPU_VGATHER_PLANNER_H_
#define _CPU_VGATHER_PLANNER_H_

#include "../../../../../modules/core/thread_management/thread_arrange.h"
#include "../../../../Element_wise/common/cpu_element_wise_planner.h"
#include "../../interpolate_types.h"
#include "vgather_addr_manager.h"
#include "../../../../Classes/type_info.h"
#include "../../interpolate_types.h"


namespace decx
{
    class cpu_VGT2D_planner;
}


class decx::cpu_VGT2D_planner : public decx::cpu_ElementWise2D_planner
{
protected:
    de::Interpolate_Types _interpolate_type;

    uint32_t _pitchsrc_v1;

public:
    cpu_VGT2D_planner() {
        memset(this, 0, sizeof(decx::cpu_VGT2D_planner));
    }


    _CRSR_
    void plan(const uint32_t concurrency, const uint2 dst_dims_v1, const uint8_t datatype_size,
        const de::Interpolate_Types intp_type, const uint2 src_dims_v1, de::DH* handle, 
        uint64_t min_thread_proc = _EW_MIN_THREAD_PROC_DEFAULT_CPU_);


    template <typename _type_in, typename _type_out>
    void run(const _type_in* src, const float2* map, _type_out* dst, const uint32_t pitchmat_v1, 
        const uint32_t pitchdst_v1, decx::utils::_thr_1D* t1D);


    static void release(decx::cpu_VGT2D_planner* fake_this);

private:
    decx::PtrInfo<void> _addr_mgrs;
};


#endif

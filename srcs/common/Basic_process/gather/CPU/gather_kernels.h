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

#ifndef _GATHER_KERNELS_H_
#define _GATHER_KERNELS_H_

#include "../../../basic.h"
#include "../../../../modules/core/thread_management/thread_pool.h"
#include "common/vgather_addr_manager.h"
#include "../../../Classes/Matrix.h"


namespace decx
{
namespace CPUK
{
    template <typename _data_type>
    using VGT2D_executor = void(const _data_type*, const float2*, _data_type*, const uint2, const uint32_t, 
        const uint32_t, const uint32_t, decx::CPUK::VGT_addr_mgr*);


    _THREAD_FUNCTION_ void gather2D_fp32_exec_bilinear(const float* src_head_ptr, const float2* __restrict map,
        float* __restrict dst, const uint2 proc_dims_v, const uint32_t Wsrc_v1, const uint32_t Wmap_v1, const uint32_t Wdst_v1,
        decx::CPUK::VGT_addr_mgr* _addr_info);
}
}


#endif

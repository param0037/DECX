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


#include "transpose2D_config.h"


namespace decx
{
    namespace blas {
        struct _transpose_profiles_bytes {
            uint8_t _alignment;
            uint16_t _block_H;
            uint16_t _block_W;
        };

        static decx::blas::_transpose_profiles_bytes _profiles[5] = {
            {8, 32, 32},    // 1
            {8, 32, 32},    // 2
            {4, 32, 32},    // 4
            {2, 32, 32},    // 8
            {2, 16, 16},    // 16
        };
    }
}


void decx::blas::_cpu_transpose_config::
config(const uint8_t _element_byte, const uint32_t concurrency, const uint2 src_dims_v1, de::DH* handle)
{
    const decx::blas::_transpose_profiles_bytes _profile = 
        decx::blas::_profiles[decx::utils::_GetHighest_abd(_element_byte)];

    this->_concurrency = concurrency;
    // Use realloc instead to allow multiple configure calling
    if (decx::alloc::_host_virtual_page_realloc(&this->_blocking_configs, 
        this->_concurrency * sizeof(decx::utils::_blocking2D_fmgrs))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    
    this->_src_proc_dims_v1 = src_dims_v1;

    // Plan the thread distribution
    decx::utils::thread2D_arrangement_advisor(&this->_thread_dist2D, this->_concurrency, this->_src_proc_dims_v1);

    decx::utils::frag_manager_gen_Nx(&this->_fmgr_H, this->_src_proc_dims_v1.y, this->_thread_dist2D.y, _profile._alignment);
    decx::utils::frag_manager_gen_Nx(&this->_fmgr_W, this->_src_proc_dims_v1.x, this->_thread_dist2D.x, _profile._alignment);
    
    uint32_t _linear_dex = 0;
    for (uint32_t i = 0; i < this->_thread_dist2D.y; ++i) 
    {
        uint2 proc_dims = make_uint2(this->_fmgr_W.frag_len, 
                                     i < this->_thread_dist2D.y - 1 ? this->_fmgr_H.frag_len : 
                                     this->_fmgr_H.last_frag_len);
        
        for (uint32_t j = 0; j < this->_thread_dist2D.x; ++j) 
        {
            if (j == this->_thread_dist2D.x - 1) {
                proc_dims.x = this->_fmgr_W.last_frag_len;
            }

            decx::utils::frag_manager_gen_from_fragLen(&this->_blocking_configs.ptr[_linear_dex]._fmgrH,
                                             proc_dims.y,
                                             _profile._block_H);

            decx::utils::frag_manager_gen_from_fragLen(&this->_blocking_configs.ptr[_linear_dex]._fmgrW,
                                             proc_dims.x,
                                             _profile._block_W);

            ++_linear_dex;
        }
    }
}


uint2 decx::blas::_cpu_transpose_config::GetThreadDist2D() const
{
    return this->_thread_dist2D;
}


void decx::blas::_cpu_transpose_config::
release(decx::blas::_cpu_transpose_config* _fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_blocking_configs);
}


bool decx::blas::_cpu_transpose_config::
changed(const uint8_t element_byte, const uint32_t concurrency, const uint2 src_dims_v1) const
{
    uint32_t _size_matched = this->_element_byte ^ element_byte;
    uint32_t _conc_matched = this->_concurrency ^ concurrency;
    uint32_t _proc_dims_matched = this->_src_proc_dims_v1.x ^ src_dims_v1.x;
    _proc_dims_matched ^= (this->_src_proc_dims_v1.y ^ src_dims_v1.y);
    return _conc_matched ^ _proc_dims_matched ^ _size_matched;
}

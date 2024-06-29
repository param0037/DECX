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


#ifndef _TRANSPOSE2D_CONFIGS_H_
#define _TRANSPOSE2D_CONFIGS_H_

#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/resources_manager/decx_resource.h"

namespace decx
{
namespace blas
{
    class _cpu_transpose_config;


    class _cpu_transpose_MK_config;


    extern decx::ResourceHandle g_cpu_transpose_4b_config;
    extern decx::ResourceHandle g_cpu_transpose_1b_config;
    extern decx::ResourceHandle g_cpu_transpose_8b_config;
    extern decx::ResourceHandle g_cpu_transpose_16b_config;
}
}


class _DECX_API_ decx::blas::_cpu_transpose_config
{
private:
    uint8_t _element_byte;
    decx::PtrInfo<decx::utils::_blocking2D_fmgrs> _blocking_configs;
    decx::utils::frag_manager _fmgr_H, _fmgr_W;
    uint2 _thread_dist2D;
    uint2 _src_proc_dims_v1;
    uint32_t _concurrency;


public:
    _cpu_transpose_config() {}


    void _CRSR_ config(const uint8_t _element_byte, const uint32_t concurrency, const uint2 src_dims_v1, de::DH* handle);


    void transpose_4b_caller(const float* src, float* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_1b_caller(const uint64_t* src, uint64_t* dst, const uint32_t pitchsrc_v8,
        const uint32_t pitchdst_v8, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_8b_caller(const double* src, double* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_16b_caller(const double* src, double* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    bool changed(const uint8_t element_byte, const uint32_t concurrency, const uint2 src_dims_v1) const;


    uint2 GetThreadDist2D() const;


    static void release(decx::blas::_cpu_transpose_config* _fake_this);
};


//
//template <uint8_t _element_byte>
//struct decx::blas::_cpu_transpose_MK_config
//{
//    decx::utils::frag_manager _f_mgr;
//    uint2 _src_proc_dims;
//    uint64_t _gapsrc_v1, _gapdst_v1;
//    uint32_t _channel_num;
//
//
//    _cpu_transpose_MK_config() {}
//
//
//    _cpu_transpose_MK_config(const uint2 _proc_dims_src, 
//                             const uint32_t _thread_num, 
//                             const uint32_t _channel_num,
//                             const uint64_t _gapsrc_v1, 
//                             const uint64_t _gapdst_v1) 
//    {
//        uint8_t _alignment = 16 / _element_byte;
//        this->_channel_num = _channel_num;
//        this->_gapsrc_v1 = _gapsrc_v1;
//        this->_gapdst_v1 = _gapdst_v1;
//
//        this->_src_proc_dims = _proc_dims_src;
//
//        if (_element_byte == 1) {
//            _alignment = 8;
//        }
//        if (_channel_num < _thread_num) {
//            decx::utils::frag_manager_gen_Nx(&this->_f_mgr, _proc_dims_src.y, _thread_num, _alignment);
//        }
//        else {
//            decx::utils::frag_manager_gen(&this->_f_mgr, this->_channel_num, _thread_num);
//        }
//    }
//
//    template <uint8_t _byte_src>
//    decx::blas::_cpu_transpose_MK_config<_element_byte>& operator=(const decx::bp::_cpu_transpose_MK_config<_byte_src>& src)
//    {
//        if ((void*)this != (void*)(&src)) {
//            this->_f_mgr = src._f_mgr;
//            this->_src_proc_dims = src._src_proc_dims;
//            this->_gapdst_v1 = src._gapdst_v1;
//            this->_gapsrc_v1 = src._gapsrc_v1;
//            this->_channel_num = src._channel_num;
//        }
//        return *this;
//    }
//};



#endif

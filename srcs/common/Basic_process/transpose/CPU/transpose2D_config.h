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

#include "../../../basic.h"
#include "../../../../modules/core/thread_management/thread_arrange.h"
#include "../../../FMGR/fragment_arrangment.h"
#include "../../../../modules/core/thread_management/thread_pool.h"
#include "../../../../modules/core/resources_manager/decx_resource.h"
#include "../../../Classes/classes_util.h"

namespace decx
{
namespace blas
{
    struct _transpose_profiles_bytes {
        uint8_t _alignment;
        uint16_t _block_H;
        uint16_t _block_W;
    };

    class _cpu_transpose_config;


    class _cpu_transpose_MC_config;


    extern decx::ResourceHandle g_cpu_transpose_4b_config;
    extern decx::ResourceHandle g_cpu_transpose_1b_config;
    extern decx::ResourceHandle g_cpu_transpose_8b_config;
    extern decx::ResourceHandle g_cpu_transpose_16b_config;
}
}


class _DECX_API_ decx::blas::_cpu_transpose_config
{
    friend class decx::blas::_cpu_transpose_MC_config;

private:
    uint8_t _element_byte;
    decx::PtrInfo<decx::utils::_blocking2D_fmgrs> _blocking_configs;
    decx::utils::frag_manager _fmgr_H, _fmgr_W;
    uint2 _thread_dist2D;
    uint2 _src_proc_dims_v1;
    uint32_t _concurrency;


    void _CRSR_ _plan_threading(const decx::blas::_transpose_profiles_bytes* _profile, de::DH* handle);


public:
    _cpu_transpose_config() {}


    void _CRSR_ config(const uint8_t _element_byte, const uint32_t concurrency, const uint2 src_dims_v1, de::DH* handle);


    void transpose_4b_caller(const float* src, float* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_1b_caller(const uint64_t* src, uint64_t* dst, const uint32_t pitchsrc_v8,
        const uint32_t pitchdst_v8, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_8b_caller(const double* src, double* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_8b_MC_caller(const double* src, double* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, const uint32_t ch_num, const uint64_t gch_src_v1, const uint64_t gch_dst_v1,
        decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_16b_caller(const de::CPd* src, de::CPd* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_16b_MC_caller(const de::CPd* src, de::CPd* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, const uint32_t ch_num, const uint64_t gch_src_v1, const uint64_t gch_dst_v1,
        decx::utils::_thread_arrange_1D* t1D) const;


    bool changed(const uint8_t element_byte, const uint32_t concurrency, const uint2 src_dims_v1) const;


    uint2 GetThreadDist2D() const;


    static void release(decx::blas::_cpu_transpose_config* _fake_this);
};


class _DECX_API_ decx::blas::_cpu_transpose_MC_config
{
private:
    uint64_t _ch_gap_src;
    uint64_t _ch_gap_dst;

    uint32_t _channel_num;
    decx::utils::frag_manager _fmgr_ch;

    decx::blas::_cpu_transpose_config _parallel_transp_config;

    bool _divide_ch;

    decx::utils::_blocking2D_fmgrs _blocking_conf;

public:
    _cpu_transpose_MC_config() {}


    void _CRSR_ config(const uint8_t _element_byte, const uint32_t concurrency, 
        const uint2 src_dims_v1, const uint32_t ch_num, const uint64_t gch_src_v1,
        const uint64_t gch_dst_v1, de::DH* handle);


    void transpose_8b_caller(const double* src, double* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;


    void transpose_16b_caller(const de::CPd* src, de::CPd* dst, const uint32_t pitchsrc_v1,
        const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const;
};



#endif

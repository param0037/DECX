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


#include "scan_caller.h"
#include "../../../modules/core/allocators.h"
#include "scan.cuh"


#define _CUDA_SCAN2D_ALIGN_2B_ 8
#define _CUDA_SCAN2D_ALIGN_4B_ 4
#define _CUDA_SCAN2D_ALIGN_1B_ 16


namespace decx
{
    namespace scan
    {
        struct cuda_scan2D_key_param_configs
        {
            /*
            * Denotes the memory alignment of source matrix and destinated matrix
            */
            uint32_t _align_src, _align_dst;

            /*
            * Packaged processed vector length
            */
            uint32_t proc_VL_H, proc_VL_V;
            uint32_t _auxiliary_proc_V;

            template <typename _type_in, typename _type_out>
            void _generate_configs(const bool _is_full_scan);
        };
    }
}


template <>
void decx::scan::cuda_scan2D_key_param_configs::_generate_configs<uint8_t, int>(const bool _is_full_scan)
{
    if (_is_full_scan) {
        this->proc_VL_H = 8;
    }
    else {
        this->proc_VL_H = 4;
        this->_auxiliary_proc_V = 4;
    }
    this->proc_VL_V = 1;

    this->_align_src = _CUDA_SCAN2D_ALIGN_1B_;
    this->_align_dst = _CUDA_SCAN2D_ALIGN_4B_;
}


template <>
void decx::scan::cuda_scan2D_key_param_configs::_generate_configs<float, float>(const bool _is_full_scan)
{
    this->proc_VL_H = 4;
    this->proc_VL_V = 1;

    this->_align_src = _CUDA_SCAN2D_ALIGN_4B_;
    this->_align_dst = _CUDA_SCAN2D_ALIGN_4B_;
}


template <>
void decx::scan::cuda_scan2D_key_param_configs::_generate_configs<de::Half, float>(const bool _is_full_scan)
{
    this->proc_VL_V = 1;
    
    if (_is_full_scan) {
        this->proc_VL_H = 8;
    }
    else {
        this->proc_VL_H = 2;
        this->_auxiliary_proc_V = 4; //1
    }

    this->_align_src = _CUDA_SCAN2D_ALIGN_2B_;
    this->_align_dst = _CUDA_SCAN2D_ALIGN_4B_;
}



template <typename _type_in, typename _type_out>
void decx::scan::cuda_scan2D_config::generate_scan_config(const uint2 _proc_dims, decx::cuda_stream* S, const int scan_mode,
    const bool _is_full_scan)
{
    this->_scan_mode = scan_mode;

    decx::scan::cuda_scan2D_key_param_configs _kp_configs;
    _kp_configs._generate_configs<_type_in, _type_out>(_is_full_scan);

    this->_dev_src._dims = make_uint2(decx::utils::ceil<uint32_t>(_proc_dims.x, _kp_configs._align_src) * _kp_configs._align_src, _proc_dims.y);
    this->_dev_dst._dims = make_uint2(decx::utils::ceil<uint32_t>(_proc_dims.x, _kp_configs._align_dst) * _kp_configs._align_dst, _proc_dims.y);

    this->_scan_h_grid.x = decx::utils::ceil<uint32_t>(this->_dev_dst._dims.x, 32 * _kp_configs.proc_VL_H);
    this->_scan_h_grid.z = 1;

    if (_is_full_scan) {
        this->_scan_h_grid.y = decx::utils::ceil<uint32_t>(this->_dev_dst._dims.y, 8);
    }
    else {
        this->_scan_h_grid.y = decx::utils::ceil<uint32_t>(this->_dev_dst._dims.y, 8 * _kp_configs._auxiliary_proc_V);
    }

    this->_scan_v_grid = dim3(decx::utils::ceil<uint32_t>(this->_dev_dst._dims.x, 32 * _kp_configs.proc_VL_V),
        decx::utils::ceil<uint32_t>(this->_dev_dst._dims.y, 32));


    this->_dev_tmp._dims = _dev_dst._dims;

    if (decx::alloc::_device_malloc(&this->_dev_src._ptr, this->_dev_src._dims.x * this->_dev_src._dims.y * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&this->_dev_dst._ptr, this->_dev_dst._dims.x * this->_dev_dst._dims.y * sizeof(_type_out), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (std::is_same<_type_in, uint8_t>::value) {
        if (decx::alloc::_device_malloc(&this->_dev_tmp._ptr, this->_dev_tmp._dims.x * this->_dev_tmp._dims.y * sizeof(de::Half), true, S)) {
            Print_Error_Message(4, DEV_ALLOC_FAIL);
            return;
        }
    }

    const uint64_t _larger_status_size = max(this->_scan_h_grid.x * this->_scan_h_grid.y,
        this->_scan_v_grid.x * this->_scan_v_grid.y);

    if (decx::alloc::_device_malloc(&this->_dev_status, _larger_status_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
}


template void decx::scan::cuda_scan2D_config::generate_scan_config<uint8_t, int>(const uint2, decx::cuda_stream*, const int, const bool);
template void decx::scan::cuda_scan2D_config::generate_scan_config<float, float>(const uint2, decx::cuda_stream*, const int, const bool);
template void decx::scan::cuda_scan2D_config::generate_scan_config<de::Half, float>(const uint2, decx::cuda_stream*, const int, const bool);



template <typename _type_in, typename _type_out>
void decx::scan::cuda_scan2D_config::generate_scan_config(decx::Ptr2D_Info<void> dev_src, decx::Ptr2D_Info<void> dev_dst, const uint2 _proc_dims,
    decx::cuda_stream* S, const int scan_mode, const bool _is_full_scan)
{
    this->_scan_mode = scan_mode;

    this->_dev_src = dev_src;
    this->_dev_dst = dev_dst;

    decx::scan::cuda_scan2D_key_param_configs _kp_configs;
    _kp_configs._generate_configs<_type_in, _type_out>(_is_full_scan);

    this->_scan_h_grid.x = decx::utils::ceil<uint32_t>(_proc_dims.x, 32 * _kp_configs.proc_VL_H);
    this->_scan_h_grid.z = 1;

    if (_is_full_scan) {
        this->_scan_h_grid.y = decx::utils::ceil<uint32_t>(_proc_dims.y, 8);
    }
    else {
        this->_scan_h_grid.y = decx::utils::ceil<uint32_t>(_proc_dims.y, 8 * _kp_configs._auxiliary_proc_V);
    }

    this->_scan_v_grid = dim3(decx::utils::ceil<uint32_t>(this->_dev_dst._dims.x, 32 * _kp_configs.proc_VL_V),
        decx::utils::ceil<uint32_t>(this->_dev_dst._dims.y, 32));
    // fuck
    if (std::is_same<_type_in, uint8_t>::value) 
    {
        this->_dev_tmp._dims = _dev_dst._dims;

        if (std::is_same<_type_in, uint8_t>::value) { 
            if (decx::alloc::_device_malloc(&this->_dev_tmp._ptr, this->_dev_tmp._dims.x * this->_dev_tmp._dims.y * sizeof(ushort), true, S)) {
                Print_Error_Message(4, DEV_ALLOC_FAIL);
                return;
            }
        }
    }

    const uint64_t _larger_status_size = max(this->_scan_h_grid.x * this->_scan_h_grid.y,
        this->_scan_v_grid.x * this->_scan_v_grid.y);

    if (decx::alloc::_device_malloc(&this->_dev_status, _larger_status_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
}

template void decx::scan::cuda_scan2D_config::generate_scan_config<float, float>(decx::Ptr2D_Info<void>, decx::Ptr2D_Info<void>, const uint2, decx::cuda_stream*, const int, const bool);
template void decx::scan::cuda_scan2D_config::generate_scan_config<de::Half, float>(decx::Ptr2D_Info<void>, decx::Ptr2D_Info<void>, const uint2, decx::cuda_stream*, const int, const bool);
template void decx::scan::cuda_scan2D_config::generate_scan_config<uint8_t, int>(decx::Ptr2D_Info<void>, decx::Ptr2D_Info<void>, const uint2, decx::cuda_stream*, const int, const bool);


int decx::scan::cuda_scan2D_config::get_scan_mode() const
{
    return this->_scan_mode;
}

decx::Ptr2D_Info<void> decx::scan::cuda_scan2D_config::get_raw_dev_ptr_src() const {
    return this->_dev_src;
}

decx::Ptr2D_Info<void> decx::scan::cuda_scan2D_config::get_raw_dev_ptr_dst() const {
    return this->_dev_dst;
}

decx::PtrInfo<void> decx::scan::cuda_scan2D_config::get_raw_dev_ptr_status() const {
    return this->_dev_status;
}

decx::Ptr2D_Info<void> decx::scan::cuda_scan2D_config::get_raw_dev_ptr_tmp() const {
    return this->_dev_tmp;
}


dim3 decx::scan::cuda_scan2D_config::get_scan_h_grid() const
{
    return this->_scan_h_grid;
}

dim3 decx::scan::cuda_scan2D_config::get_scan_v_grid() const
{
    return this->_scan_v_grid;
}


template <typename _Type_src>
void decx::scan::cuda_scan2D_config::release_buffer(const bool _refer_dev_classes)
{
    if (!_refer_dev_classes) {
        decx::alloc::_device_dealloc(&this->_dev_src._ptr);
        decx::alloc::_device_dealloc(&this->_dev_dst._ptr);
    }
    decx::alloc::_device_dealloc(&this->_dev_status);
    if (std::is_same<_Type_src, uint8_t>::value) {
        decx::alloc::_device_dealloc(&this->_dev_tmp._ptr);
    }
}

template void decx::scan::cuda_scan2D_config::release_buffer<float>(const bool _have_dev_classes);
template void decx::scan::cuda_scan2D_config::release_buffer<de::Half>(const bool _have_dev_classes);
template void decx::scan::cuda_scan2D_config::release_buffer<uint8_t>(const bool _have_dev_classes);


template <bool _only_scan_h>
void decx::scan::cuda_scan2D_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S)
{
    decx::Ptr2D_Info<void> src_ptr2D_info = _config->get_raw_dev_ptr_src();
    decx::PtrInfo<void> status_ptr2D_info = _config->get_raw_dev_ptr_status();
    decx::Ptr2D_Info<void> dst_ptr2D_info = _config->get_raw_dev_ptr_dst();
    
    dim3 scan_h_grid = _config->get_scan_h_grid();
    dim3 scan_v_grid = _config->get_scan_v_grid();

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        // horizontally scan
        decx::scan::GPUK::cu_h_block_exclusive_scan_fp32_2D << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)src_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr,
                                             (float4*)dst_ptr2D_info._ptr.ptr,
                                             src_ptr2D_info._dims.x / 4,
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_h_scan_DLB_fp32_2D<true> << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float4*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x / 4, 
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        if (!_only_scan_h){
        // vertically scan
        decx::scan::GPUK::cu_v_block_exclusive_scan_fp32_2D<true> << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > (NULL,
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        }
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        // horizontally scan
        decx::scan::GPUK::cu_h_block_inclusive_scan_fp32_2D << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)src_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr,
                                             (float4*)dst_ptr2D_info._ptr.ptr,
                                             src_ptr2D_info._dims.x / 4,
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_h_scan_DLB_fp32_2D<true> << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float4*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x / 4, 
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        if (!_only_scan_h) {
        // vertically scan
        decx::scan::GPUK::cu_v_block_inclusive_scan_fp32_2D<true> << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > (NULL,
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        }
        break;
    default:
        break;
    }
}

template void decx::scan::cuda_scan2D_fp32_caller_Async<true>(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);
template void decx::scan::cuda_scan2D_fp32_caller_Async<false>(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);


void decx::scan::cuda_scan2D_v_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S)
{
    decx::Ptr2D_Info<void> src_ptr2D_info = _config->get_raw_dev_ptr_src();
    decx::PtrInfo<void> status_ptr2D_info = _config->get_raw_dev_ptr_status();
    decx::Ptr2D_Info<void> dst_ptr2D_info = _config->get_raw_dev_ptr_dst();
    
    dim3 scan_v_grid = _config->get_scan_v_grid();

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        // vertically scan
        decx::scan::GPUK::cu_v_block_exclusive_scan_fp32_2D<false> << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float*)src_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        // vertically scan
        decx::scan::GPUK::cu_v_block_inclusive_scan_fp32_2D<false> << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float*)src_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        break;
    default:
        break;
    }
}


template <bool _only_scan_h>
void decx::scan::cuda_scan2D_u8_i32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S)
{
    decx::Ptr2D_Info<void> src_ptr2D_info = _config->get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> tmp_ptr2D_info = _config->get_raw_dev_ptr_tmp();
    decx::Ptr2D_Info<void> dst_ptr2D_info = _config->get_raw_dev_ptr_dst();

    dim3 scan_h_grid = _config->get_scan_h_grid();
    dim3 scan_v_grid = _config->get_scan_v_grid();

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        // horizontally scan
        decx::scan::GPUK::cu_h_block_exclusive_scan_u8_u16_2D << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float2*)src_ptr2D_info._ptr.ptr,
                                             (float4*)_config->get_raw_dev_ptr_status().ptr, 
                                             (float4*)tmp_ptr2D_info._ptr.ptr,
                                             src_ptr2D_info._dims.x / 8, 
                                             dst_ptr2D_info._dims.x / 8,
                                             scan_h_grid.x, 
                                             make_uint3(src_ptr2D_info._dims.y, 
                                                        src_ptr2D_info._dims.x / 8,
                                                        dst_ptr2D_info._dims.x / 8));

        // horizontally decoupled lookback
        decx::scan::GPUK::cu_h_block_DLB_fp16_i32_2D_v8<true> << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)tmp_ptr2D_info._ptr.ptr,
                                             (float4*)_config->get_raw_dev_ptr_status().ptr,
                                             (int4*)dst_ptr2D_info._ptr.ptr,
                                             dst_ptr2D_info._dims.x / 4, 
                                             dst_ptr2D_info._dims.x / 8, 
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 8, dst_ptr2D_info._dims.y));
        if (!_only_scan_h) {
        // vertically scan
        decx::scan::GPUK::cu_v_block_exclusive_scan_int32_2D << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status().ptr, 
                                             (int*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        // vertically ddecoupled lookback
        decx::scan::GPUK::cu_v_scan_DLB_int32_2D<true> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status().ptr,
                                             (int*)dst_ptr2D_info._ptr.ptr,
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        }
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:

        // horizontally scan
        decx::scan::GPUK::cu_h_block_inclusive_scan_u8_u16_2D << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float2*)src_ptr2D_info._ptr.ptr,
                                             (float4*)_config->get_raw_dev_ptr_status().ptr, 
                                             (float4*)tmp_ptr2D_info._ptr.ptr,
                                             src_ptr2D_info._dims.x / 8, 
                                             dst_ptr2D_info._dims.x / 8,
                                             scan_h_grid.x, 
                                             make_uint3(src_ptr2D_info._dims.y, 
                                                        src_ptr2D_info._dims.x / 8,
                                                        dst_ptr2D_info._dims.x / 8));

        // horizontally decoupled lookback
        decx::scan::GPUK::cu_h_block_DLB_fp16_i32_2D_v8<false> << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)tmp_ptr2D_info._ptr.ptr,
                                             (float4*)_config->get_raw_dev_ptr_status().ptr,
                                             (int4*)dst_ptr2D_info._ptr.ptr,
                                             dst_ptr2D_info._dims.x / 4, 
                                             dst_ptr2D_info._dims.x / 8, 
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 8, dst_ptr2D_info._dims.y));

        if (!_only_scan_h) {
        // vertically scan
        decx::scan::GPUK::cu_v_block_inclusive_scan_int32_2D << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status().ptr, 
                                             (int*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        // vertically ddecoupled lookback
        decx::scan::GPUK::cu_v_scan_DLB_int32_2D<false> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status().ptr,
                                             (int*)dst_ptr2D_info._ptr.ptr,
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        }
        break;
    default:
        break;
    }
}


template void decx::scan::cuda_scan2D_u8_i32_caller_Async<true>(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);
template void decx::scan::cuda_scan2D_u8_i32_caller_Async<false>(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);


template <bool _only_scan_h>
void decx::scan::cuda_scan2D_fp16_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S)
{
    decx::Ptr2D_Info<void> src_ptr2D_info = _config->get_raw_dev_ptr_src();
    decx::PtrInfo<void> status_ptr2D_info = _config->get_raw_dev_ptr_status();
    decx::Ptr2D_Info<void> dst_ptr2D_info = _config->get_raw_dev_ptr_dst();

    dim3 scan_h_grid = _config->get_scan_h_grid();
    dim3 scan_v_grid = _config->get_scan_v_grid();

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        // horizontally scan
        decx::scan::GPUK::cu_h_block_exclusive_scan_fp16_2D << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)src_ptr2D_info._ptr.ptr, 
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float4*)dst_ptr2D_info._ptr.ptr, 
                                             src_ptr2D_info._dims.x / 8, 
                                             dst_ptr2D_info._dims.x / 4,
                                             scan_h_grid.x, 
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_h_scan_DLB_fp32_2D_v8<true> << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float4*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x / 4, 
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 8, dst_ptr2D_info._dims.y));

        if (!_only_scan_h) {
        // vertically scan
        decx::scan::GPUK::cu_v_block_exclusive_scan_fp32_2D<true> << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > (NULL,
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        }
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        // horizontally scan
        decx::scan::GPUK::cu_h_block_inclusive_scan_fp16_2D << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)src_ptr2D_info._ptr.ptr, 
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float4*)dst_ptr2D_info._ptr.ptr, 
                                             src_ptr2D_info._dims.x / 8, 
                                             dst_ptr2D_info._dims.x / 4,
                                             scan_h_grid.x, 
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_h_scan_DLB_fp32_2D_v8<false> << <scan_h_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float4*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x / 4, 
                                             scan_h_grid.x,
                                             make_uint2(dst_ptr2D_info._dims.x / 8, dst_ptr2D_info._dims.y));

        if (!_only_scan_h) {
        // vertically scan
        decx::scan::GPUK::cu_v_block_inclusive_scan_fp32_2D<true> << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > (NULL,
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<false> << <scan_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        }
        break;
    default:
        break;
    }
}

template void decx::scan::cuda_scan2D_fp16_fp32_caller_Async<true>(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);
template void decx::scan::cuda_scan2D_fp16_fp32_caller_Async<false>(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);



void decx::scan::cuda_scan2D_v_fp16_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S)
{
    decx::Ptr2D_Info<void> src_ptr2D_info = _config->get_raw_dev_ptr_src();
    decx::PtrInfo<void> status_ptr2D_info = _config->get_raw_dev_ptr_status();
    decx::Ptr2D_Info<void> dst_ptr2D_info = _config->get_raw_dev_ptr_dst();

    dim3 scan_v_grid = _config->get_scan_h_grid();
    dim3 DLB_v_grid = _config->get_scan_v_grid();

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        // horizontally scan
        decx::scan::GPUK::cu_v_block_exclusive_scan_fp16_2D_v2 << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float*)src_ptr2D_info._ptr.ptr, 
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float2*)dst_ptr2D_info._ptr.ptr, 
                                             src_ptr2D_info._dims.x / 2, 
                                             dst_ptr2D_info._dims.x / 2, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x / 2, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true> << <DLB_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        // vertically scan
        decx::scan::GPUK::cu_v_block_inclusive_scan_fp16_2D_v2 << < scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float*)src_ptr2D_info._ptr.ptr, 
                                             (float4*)status_ptr2D_info.ptr, 
                                             (float2*)dst_ptr2D_info._ptr.ptr, 
                                             src_ptr2D_info._dims.x / 2, 
                                             dst_ptr2D_info._dims.x / 2, 
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x / 2, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<false> << <DLB_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((float4*)status_ptr2D_info.ptr, 
                                             (float*)dst_ptr2D_info._ptr.ptr, 
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        break;

    default:
        break;
    }
}



void decx::scan::cuda_scan2D_v_u8_i32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S)
{
    decx::Ptr2D_Info<void> src_ptr2D_info = _config->get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> tmp_ptr2D_info = _config->get_raw_dev_ptr_tmp();
    decx::PtrInfo<void> status_ptr2D_info = _config->get_raw_dev_ptr_status();
    decx::Ptr2D_Info<void> dst_ptr2D_info = _config->get_raw_dev_ptr_dst();

    dim3 scan_v_grid = _config->get_scan_h_grid();
    dim3 DLB_v_grid = _config->get_scan_v_grid();

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        decx::scan::GPUK::cu_v_block_inclusive_scan_u8_u16_2D_v4 << <scan_v_grid, dim3(32, 8), 
            0, S->get_raw_stream_ref() >> > ((float*)src_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr,
                                             (double*)tmp_ptr2D_info._ptr.ptr,
                                             src_ptr2D_info._dims.x / 4,
                                             tmp_ptr2D_info._dims.x / 4,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_u16_i32_2D<false><< <DLB_v_grid, dim3(32, 32), 
            0, S->get_raw_stream_ref() >> > ((ushort*)tmp_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr,
                                             (int*)dst_ptr2D_info._ptr.ptr,
                                             tmp_ptr2D_info._dims.x,
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        decx::scan::GPUK::cu_v_block_exclusive_scan_u8_u16_2D_v4 << <scan_v_grid, dim3(32, 8),
            0, S->get_raw_stream_ref() >> > ((float*)src_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr,
                                             (double*)tmp_ptr2D_info._ptr.ptr,
                                             src_ptr2D_info._dims.x / 4,
                                             tmp_ptr2D_info._dims.x / 4,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x / 4, dst_ptr2D_info._dims.y));

        decx::scan::GPUK::cu_v_scan_DLB_u16_i32_2D<true> << <DLB_v_grid, dim3(32, 32),
            0, S->get_raw_stream_ref() >> > ((ushort*)tmp_ptr2D_info._ptr.ptr,
                                             (float4*)status_ptr2D_info.ptr,
                                             (int*)dst_ptr2D_info._ptr.ptr,
                                             tmp_ptr2D_info._dims.x,
                                             dst_ptr2D_info._dims.x,
                                             scan_v_grid.y,
                                             make_uint2(dst_ptr2D_info._dims.x, dst_ptr2D_info._dims.y));
        break;
    default:
        break;
    }
}


// undefine to prevent macro leakage
#ifdef _CUDA_SCAN2D_ALIGN_1B_
#undef _CUDA_SCAN2D_ALIGN_1B_
#endif
#ifdef _CUDA_SCAN2D_ALIGN_2B_
#undef _CUDA_SCAN2D_ALIGN_2B_
#endif
#ifdef _CUDA_SCAN2D_ALIGN_4B_
#undef _CUDA_SCAN2D_ALIGN_4B_
#endif
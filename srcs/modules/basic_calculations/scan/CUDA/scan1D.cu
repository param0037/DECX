/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "scan.cuh"
#include "scan_caller.h"
#include "../../../core/allocators.h"



template <uint32_t _align, typename _type_in, typename _type_out>
void decx::scan::cuda_scan1D_config::generate_scan_config(const uint64_t _proc_length, decx::cuda_stream* S, const int scan_mode)
{
    this->_scan_mode = scan_mode;

    this->_length = decx::utils::ceil<uint64_t>(_proc_length, _align) * _align;

    this->_block_num = decx::utils::ceil<uint64_t>(this->_length / _align, _WARP_SCAN_BLOCK_SIZE_);

    if (decx::alloc::_device_malloc(&this->_dev_dst, this->_length * sizeof(_type_out), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_dev_tmp, this->_length * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }


    if (decx::alloc::_device_malloc(&this->_dev_status, this->_block_num * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_dev_src, this->_length * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
}


template void decx::scan::cuda_scan1D_config::generate_scan_config<8, de::Half, float>(const uint64_t, decx::cuda_stream*, const int);
template void decx::scan::cuda_scan1D_config::generate_scan_config<4, float>(const uint64_t, decx::cuda_stream*, const int);
template void decx::scan::cuda_scan1D_config::generate_scan_config<8, uint8_t, int>(const uint64_t, decx::cuda_stream*, const int);


template <uint32_t _align, typename _type_in, typename _type_out>
void decx::scan::cuda_scan1D_config::generate_scan_config(decx::PtrInfo<void>   dev_src, 
                                                          decx::PtrInfo<void>   dev_dst, 
                                                          const uint64_t        _proc_length,
                                                          decx::cuda_stream*    S, 
                                                          const int             scan_mode)
{
    this->_scan_mode = scan_mode;

    this->_length = decx::utils::ceil<uint64_t>(_proc_length, _align) * _align;

    this->_block_num = decx::utils::ceil<uint64_t>(this->_length / _align, _WARP_SCAN_BLOCK_SIZE_);

    this->_dev_dst = dev_dst;
    this->_dev_src = dev_src;
    if (std::is_same<_type_in, uint8_t>::value) {
        if (decx::alloc::_device_malloc(&this->_dev_tmp, this->_length * sizeof(de::Half), true, S)) {
            Print_Error_Message(4, DEV_ALLOC_FAIL);
            return;
        }
    }

    if (decx::alloc::_device_malloc(&this->_dev_status, this->_block_num * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
}

template void decx::scan::cuda_scan1D_config::generate_scan_config<8, de::Half, float>(decx::PtrInfo<void>, decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*, const int);
template void decx::scan::cuda_scan1D_config::generate_scan_config<4, float, float>(decx::PtrInfo<void>, decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*, const int);
template void decx::scan::cuda_scan1D_config::generate_scan_config<8, uint8_t, int>(decx::PtrInfo<void>, decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*, const int);



uint64_t decx::scan::cuda_scan1D_config::get_proc_length() const
{
    return this->_length;
}

uint64_t decx::scan::cuda_scan1D_config::get_block_num() const
{
    return this->_block_num;
}

int decx::scan::cuda_scan1D_config::get_scan_mode() const
{
    return this->_scan_mode;
}

void* decx::scan::cuda_scan1D_config::get_raw_dev_ptr_src() const {
    return this->_dev_src.ptr;
}

void* decx::scan::cuda_scan1D_config::get_raw_dev_ptr_dst() const {
    return this->_dev_dst.ptr;
}

void* decx::scan::cuda_scan1D_config::get_raw_dev_ptr_status() const {
    return this->_dev_status.ptr;
}


void* decx::scan::cuda_scan1D_config::get_raw_dev_ptr_tmp() const {
    return this->_dev_tmp.ptr;
}



template <typename _src_type>
void decx::scan::cuda_scan1D_config::release_buffer(const bool _have_dev_classes)
{
    if (_have_dev_classes) {
        decx::alloc::_device_dealloc(&this->_dev_src);
        decx::alloc::_device_dealloc(&this->_dev_dst);
    }
    if (std::is_same<_src_type, uint8_t>::value) {
        decx::alloc::_device_dealloc(&this->_dev_status);
    }
}

template void decx::scan::cuda_scan1D_config::release_buffer<float>(const bool _have_dev_classes);
template void decx::scan::cuda_scan1D_config::release_buffer<de::Half>(const bool _have_dev_classes);
template void decx::scan::cuda_scan1D_config::release_buffer<uint8_t>(const bool _have_dev_classes);


void decx::scan::cuda_scan1D_fp32_caller_Async(const decx::scan::cuda_scan1D_config* _config, decx::cuda_stream* S)
{
    const uint64_t length_v4 = _config->get_proc_length() / 4;
    
    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        decx::scan::GPUK::cu_block_exclusive_scan_fp32_1D << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_src(),
                (float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v4);

        decx::scan::GPUK::cu_scan_DLB_fp32_1D<true> << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v4);
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        decx::scan::GPUK::cu_block_inclusive_scan_fp32_1D << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_src(),
                (float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v4);

        decx::scan::GPUK::cu_scan_DLB_fp32_1D<false> << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v4);
        break;
    default:
        break;
    }
}


void decx::scan::cuda_scan1D_u8_i32_caller_Async(const decx::scan::cuda_scan1D_config* _config, decx::cuda_stream* S)
{
    const uint64_t length_v8 = _config->get_proc_length() / 8;     // / 16
    
    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        decx::scan::GPUK::cu_block_exclusive_scan_u8_fp16_1D << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float2*)_config->get_raw_dev_ptr_src(),
                                             (float4*)_config->get_raw_dev_ptr_status(),
                                             (int4*)_config->get_raw_dev_ptr_tmp(),
                                             length_v8);

        decx::scan::GPUK::cu_block_DLB_u16_i32_1D_v8<true> << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_tmp(),
                                             (float4*)_config->get_raw_dev_ptr_status(),
                                             (int4*)_config->get_raw_dev_ptr_dst(),
                                             _config->get_proc_length() / 8);
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        decx::scan::GPUK::cu_block_inclusive_scan_u8_u16_1D << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float2*)_config->get_raw_dev_ptr_src(),
                                             (float4*)_config->get_raw_dev_ptr_status(),
                                             (int4*)_config->get_raw_dev_ptr_tmp(),
                                             length_v8);

        decx::scan::GPUK::cu_block_DLB_u16_i32_1D_v8<false> << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_tmp(),
                                             (float4*)_config->get_raw_dev_ptr_status(),
                                             (int4*)_config->get_raw_dev_ptr_dst(),
                                             _config->get_proc_length() / 8);
        break;
    default:
        break;
    }
}



void decx::scan::cuda_scan1D_fp16_caller_Async(const decx::scan::cuda_scan1D_config* _config, decx::cuda_stream* S)
{
    const uint64_t length_v8 = _config->get_proc_length() / 8;

    switch (_config->get_scan_mode())
    {
    case decx::scan::SCAN_MODE::SCAN_MODE_EXCLUSIVE:
        decx::scan::GPUK::cu_block_exclusive_scan_fp16_1D << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_src(),
                (float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v8);

        decx::scan::GPUK::cu_scan_DLB_fp32_1D_v8<true> << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v8);
        break;

    case decx::scan::SCAN_MODE::SCAN_MODE_INCLUSIVE:
        decx::scan::GPUK::cu_block_inclusive_scan_fp16_1D << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_src(),
                (float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v8);

        decx::scan::GPUK::cu_scan_DLB_fp32_1D_v8<false> << <_config->get_block_num(), _WARP_SCAN_BLOCK_SIZE_,
            0, S->get_raw_stream_ref() >> > ((float4*)_config->get_raw_dev_ptr_status(),
                (float4*)_config->get_raw_dev_ptr_dst(),
                length_v8);
        break;
    default:
        break;
    }
}
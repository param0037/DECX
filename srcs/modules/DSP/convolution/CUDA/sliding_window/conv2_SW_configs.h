/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_SW_CONFIGS_H_
#define _CONV2_SW_CONFIGS_H_


#include "../../conv_utils.h"


namespace decx
{
    namespace conv
    {
        struct _cuConv2_kernel_params 
        {
            _matrix_configs _src_confs, _kernel_confs, _dst_confs;
            uint2 src_buf_dims, ker_dims, dst_dims, kernel_shift, cpy_shift;
        };


        template <typename _data_type>
        class _cuda_conv2_preset;


        typedef _cuda_conv2_preset<float> _cuda_conv2_fp32_preset;
        typedef _cuda_conv2_preset<de::Half> _cuda_conv2_fp16_preset;
        typedef _cuda_conv2_preset<uint8_t> _cuda_conv2_uc8_uc8_preset;


        template <uint _bound_H, uint _bound_W> static void
            _cuda_conv2_fp32_NB_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params);


        template <uint _bound_H, uint _bound_W> static void
            _cuda_conv2_fp32_BC_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params);


        template <uint _bound_H, uint _bound_W> static void
            _cuda_conv2_fp16_NB_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params);


        template <uint _bound_H, uint _bound_W> static void
            _cuda_conv2_fp16_BC_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params);

        /*
        * Should be called before k_params->src_confs, k_params->kernel_confs and k_params->dst_confs
        * are all configured !
        */
        static void _cuda_conv2_uint8_NB_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params);

        /*
        * Should be called before k_params->src_confs, k_params->kernel_confs and k_params->dst_confs
        * are all configured !
        */
        static void _cuda_conv2_uint8_BC_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params);
    }
}



template <typename _data_type>
class decx::conv::_cuda_conv2_preset
{
public:
    decx::PtrInfo<void> src_buf;
    decx::PtrInfo<void> ker_buf;
    decx::PtrInfo<void> dst_buf;
    decx::conv::_cuConv2_kernel_params _Kparams;
    decx::cuda_stream* S;
    decx::cuda_event* E;
    cudaMemcpyKind memcpy_flag;

    template <bool _print, typename _dst_type, typename _kernel_type>
    void _cuda_conv2_malloc(de::DH* handle);


    template <typename _kernel_type>
    void _cuda_conv2_memcpyH2D();


    void _cuda_conv2_MC_src_memcpy_from_host(const uint _mat_id);


    template <bool _is_MK, typename _kernel_type>
    void _cuda_conv2_MC_kernel_memcpy_from_host(const uint _mat_id = 0);


    template <typename _dst_type>
    void _cuda_conv2_memcpyD2H();


    void release();
};




template <typename _data_type>
template <bool _print, typename _dst_type, typename _kernel_type>
void decx::conv::_cuda_conv2_preset<_data_type>::_cuda_conv2_malloc(de::DH* handle)
{
    if (decx::alloc::_device_malloc(&this->src_buf,
        this->_Kparams.src_buf_dims.x * this->_Kparams.src_buf_dims.y * sizeof(_data_type), true, this->S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&this->ker_buf,
        this->_Kparams.ker_dims.x * this->_Kparams.ker_dims.y * sizeof(_kernel_type), true, this->S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    if (this->memcpy_flag != cudaMemcpyDeviceToDevice) {
        if (decx::alloc::_device_malloc(&this->dst_buf,
            this->_Kparams.dst_dims.x * this->_Kparams.dst_dims.y * sizeof(_dst_type), true, this->S)) {
            decx::err::device_AllocateFailure<_print>(handle);
            return;
        }
    }
}



template <typename _data_type>
template <typename _kernel_type>
void decx::conv::_cuda_conv2_preset<_data_type>::_cuda_conv2_memcpyH2D()
{
    decx::conv::_cuConv2_kernel_params* k_params = &this->_Kparams;

    checkCudaErrors(cudaMemcpy2DAsync(this->ker_buf.ptr,
        k_params->ker_dims.x * sizeof(_kernel_type),
        k_params->_kernel_confs._ptr,
        k_params->_kernel_confs._pitch * sizeof(_kernel_type),
        k_params->ker_dims.x * sizeof(_kernel_type),
        k_params->ker_dims.y,
        this->memcpy_flag,
        this->S->get_raw_stream_ref()));

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<_data_type*>(this->src_buf.ptr) + this->_Kparams.cpy_shift.x * k_params->src_buf_dims.x + this->_Kparams.cpy_shift.y,
        k_params->src_buf_dims.x * sizeof(_data_type),
        k_params->_src_confs._ptr,
        k_params->_src_confs._pitch * sizeof(_data_type),
        k_params->_src_confs._width * sizeof(_data_type),
        k_params->_src_confs._height,
        this->memcpy_flag,
        this->S->get_raw_stream_ref()));                            // copy the datas of src from host to device
}




template <typename _data_type>
void decx::conv::_cuda_conv2_preset<_data_type>::_cuda_conv2_MC_src_memcpy_from_host(const uint _mat_id)
{
    decx::conv::_cuConv2_kernel_params* k_params = &this->_Kparams;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<_data_type*>(this->src_buf.ptr) + this->_Kparams.cpy_shift.x * k_params->src_buf_dims.x + this->_Kparams.cpy_shift.y,
        k_params->src_buf_dims.x * sizeof(_data_type),
        k_params->_src_confs._ptr_array[_mat_id],
        k_params->_src_confs._pitch * sizeof(_data_type),
        k_params->_src_confs._width * sizeof(_data_type),
        k_params->_src_confs._height,
        this->memcpy_flag,
        this->S->get_raw_stream_ref()));                            // copy the datas of src from host to device
}




template <typename _data_type>
template <bool _is_MK, typename _kernel_type>
void decx::conv::_cuda_conv2_preset<_data_type>::_cuda_conv2_MC_kernel_memcpy_from_host(const uint _mat_id)
{
    decx::conv::_cuConv2_kernel_params* k_params = &this->_Kparams;

    void* kernel_ptr = _is_MK ? k_params->_kernel_confs._ptr_array[_mat_id] : k_params->_kernel_confs._ptr;

    checkCudaErrors(cudaMemcpy2DAsync(this->ker_buf.ptr,
        k_params->ker_dims.x * sizeof(_kernel_type),
        kernel_ptr,
        k_params->_kernel_confs._pitch * sizeof(_kernel_type),
        k_params->ker_dims.x * sizeof(_kernel_type),
        k_params->ker_dims.y,
        this->memcpy_flag,
        this->S->get_raw_stream_ref()));
}





template <typename _data_type>
template <typename _dst_type>
void decx::conv::_cuda_conv2_preset<_data_type>::_cuda_conv2_memcpyD2H()
{
    const size_t _cpy_size = this->_Kparams.dst_dims.x * this->_Kparams.dst_dims.y * sizeof(_dst_type);

    checkCudaErrors(cudaMemcpyAsync(this->_Kparams._dst_confs._ptr, this->dst_buf.ptr,
        _cpy_size, cudaMemcpyDeviceToHost, this->S->get_raw_stream_ref()));
}



template <typename _data_type>
void decx::conv::_cuda_conv2_preset<_data_type>::release()
{
    decx::alloc::_device_dealloc(&this->src_buf);
    decx::alloc::_device_dealloc(&this->ker_buf);
    decx::alloc::_device_dealloc(&this->dst_buf);
    this->S->detach();
    this->E->detach();
}



template <uint _bound_H, uint _bound_W> static void
decx::conv::_cuda_conv2_fp32_NB_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params)
{
    k_params->dst_dims = make_uint2(k_params->_dst_confs._pitch, k_params->_dst_confs._height);
    k_params->src_buf_dims = make_uint2(decx::utils::ceil<uint>(k_params->_dst_confs._width, 4 * conv2_bld) * 4 * conv2_bld + _bound_W * 2,
        decx::utils::ceil<uint>(k_params->_dst_confs._height, conv2_bld) * conv2_bld + _bound_H * 2);

    k_params->ker_dims = make_uint2(k_params->_kernel_confs._width,
                                    k_params->_kernel_confs._height);

    k_params->kernel_shift.x = _bound_H - k_params->ker_dims.y / 2;                
    k_params->kernel_shift.y = _bound_W - k_params->ker_dims.x / 2;

    k_params->cpy_shift = k_params->kernel_shift;
}



template <uint _bound_H, uint _bound_W> static void
decx::conv::_cuda_conv2_fp32_BC_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params)
{
    k_params->dst_dims = make_uint2(k_params->_src_confs._pitch,
        k_params->_src_confs._height);

    k_params->src_buf_dims = make_uint2(decx::utils::ceil<uint>(k_params->_src_confs._width, 4 * conv2_bld) * 4 * conv2_bld + _bound_W * 2,
        decx::utils::ceil<uint>(k_params->_src_confs._height, conv2_bld) * conv2_bld + _bound_H * 2);

    k_params->ker_dims = make_uint2(k_params->_kernel_confs._width,
        k_params->_kernel_confs._height);

    k_params->kernel_shift.x = _bound_H - k_params->ker_dims.y / 2;                
    k_params->kernel_shift.y = _bound_W - k_params->ker_dims.x / 2;

    k_params->cpy_shift = make_uint2(_bound_H, _bound_W);
}



template <uint _bound_H, uint _bound_W> static void
decx::conv::_cuda_conv2_fp16_NB_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params)
{
    k_params->dst_dims = make_uint2(k_params->_dst_confs._pitch, k_params->_dst_confs._height);
    k_params->src_buf_dims = make_uint2(decx::utils::ceil<uint>(k_params->_dst_confs._width, 8 * conv2_bld) * 8 * conv2_bld + _bound_W * 2,
        decx::utils::ceil<uint>(k_params->_dst_confs._height, conv2_bld) * conv2_bld + _bound_H * 2);

    k_params->ker_dims = make_uint2(k_params->_kernel_confs._width,
        k_params->_kernel_confs._height);

    k_params->kernel_shift.x = _bound_H - k_params->ker_dims.y / 2;
    k_params->kernel_shift.y = _bound_W - k_params->ker_dims.x / 2;

    k_params->cpy_shift = k_params->kernel_shift;
}



template <uint _bound_H, uint _bound_W> static void
decx::conv::_cuda_conv2_fp16_BC_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params)
{
    k_params->dst_dims = make_uint2(k_params->_src_confs._pitch, k_params->_src_confs._height);

    k_params->src_buf_dims = make_uint2(decx::utils::ceil<uint>(k_params->_src_confs._width, 8 * conv2_bld) * 8 * conv2_bld + _bound_W * 2,
        decx::utils::ceil<uint>(k_params->_src_confs._height, conv2_bld) * conv2_bld + _bound_H * 2);

    k_params->ker_dims = make_uint2(k_params->_kernel_confs._width,
        k_params->_kernel_confs._height);

    k_params->kernel_shift.x = _bound_H - k_params->ker_dims.y / 2;                
    k_params->kernel_shift.y = _bound_W - k_params->ker_dims.x / 2;

    k_params->cpy_shift = make_uint2(_bound_H, _bound_W);
}



static void
decx::conv::_cuda_conv2_uint8_NB_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params)
{
    k_params->ker_dims = make_uint2(k_params->_kernel_confs._width,
                                    k_params->_kernel_confs._height);

    const uint half_kerH = k_params->ker_dims.y / 2;
    const uint half_kerW = k_params->ker_dims.x / 2;

    k_params->dst_dims = make_uint2(k_params->_dst_confs._pitch, k_params->_dst_confs._height);
    k_params->src_buf_dims.x = decx::utils::ceil<uint>(k_params->_src_confs._width, 128) * 128 + 64;

    if (half_kerH > bounded_kernel_R8) {
        k_params->src_buf_dims.y = decx::utils::ceil<uint>(k_params->_src_confs._height, 16) * 16;
    }
    else {
        k_params->src_buf_dims.y = decx::utils::ceil<uint>(k_params->_src_confs._height, 16) * 16 + 16;
    }

    k_params->kernel_shift = make_uint2(bounded_kernel_R8 - half_kerH,
                                        ((bounded_kernel_R64 - half_kerW) / 8) * 8);

    k_params->cpy_shift = make_uint2(half_kerH > bounded_kernel_R8 ? 0 : (bounded_kernel_R8 - half_kerH),
        k_params->kernel_shift.y);
}



static void
decx::conv::_cuda_conv2_uint8_BC_buf_dims_config(decx::conv::_cuConv2_kernel_params* k_params)
{
    k_params->ker_dims = make_uint2(k_params->_kernel_confs._width,
                                    k_params->_kernel_confs._height);
    const uint half_kerH = k_params->ker_dims.y / 2;
    const uint half_kerW = k_params->ker_dims.x / 2;

    k_params->dst_dims = make_uint2(k_params->_dst_confs._pitch, k_params->_src_confs._height);
    k_params->src_buf_dims.x = decx::utils::ceil<uint>(k_params->_src_confs._width + half_kerW * 2, 128) * 128 + 64;

    if (half_kerH > bounded_kernel_R8){
        k_params->src_buf_dims.y = decx::utils::ceil<uint>(k_params->_src_confs._height + half_kerH * 2, 16) * 16;
    }
    else {
        k_params->src_buf_dims.y = decx::utils::ceil<uint>(k_params->_src_confs._height + half_kerH * 2, 16) * 16 + 16;
    }

    k_params->kernel_shift = make_uint2(bounded_kernel_R8 - k_params->ker_dims.y / 2,
        ((bounded_kernel_R64 - k_params->ker_dims.x / 2) / 8) * 8);

    k_params->cpy_shift = make_uint2(half_kerH > bounded_kernel_R8 ? half_kerH : bounded_kernel_R8,
        k_params->kernel_shift.y + k_params->ker_dims.x / 2);
}



namespace decx
{
    namespace conv {
        template <typename _data_type>
        class _cuda_conv2_async_controller;

        typedef decx::conv::_cuda_conv2_async_controller<uint8_t> _CCAC_uint8;
        typedef decx::conv::_cuda_conv2_async_controller<float> _CCAC_fp32;
        typedef decx::conv::_cuda_conv2_async_controller<de::Half> _CCAC_fp16;
    }
}



template <typename _data_type>
class decx::conv::_cuda_conv2_async_controller
{
public:
    decx::conv::_cuConv2_kernel_params _Kparams;

    decx::PtrInfo<void> src_buf[2];
    decx::PtrInfo<void> ker_buf[2];
    decx::PtrInfo<void> dst_buf[2];

    decx::cuda_stream* S[3];
    decx::cuda_event* E[3];

    decx::alloc::MIF<void> dev_src[2];
    decx::alloc::MIF<void> dev_dst[2];


    void synchronize_among_all_events();


    template <bool _print>
    void generate_stream_and_event(de::DH* handle);


    template <bool _print, int src_unit_len, int dst_unit_len, int ker_unit_len>
    void _conv2_MK_alloc(de::DH* handle);


    template <uint stream_id, bool _is_MK, typename _kernel_type>
    void memcpy_src_async_H2D(const uint _mat_id);


    template <uint stream_id, bool _is_MK, typename _kernel_type>
    void memcpy_src_async_D2D(const uint _mat_id);


    template <uint stream_id, typename _kernel_type>
    void memcpy_kernel_async_H2D(const uint _mat_id);


    template <uint stream_id, typename _dst_type>
    void memcpy_dst_async_D2H(const uint _mat_id);


    template <bool _is_MK>
    void get_kernel_read_write_ptr(void** read_ptr, void** write_ptr, void** kernel_ptr = NULL);


    // Should be called after the synchronization
    void after_kernel();


    void release();
};



template <typename _data_type>
void decx::conv::_cuda_conv2_async_controller<_data_type>::synchronize_among_all_events()
{
    this->E[0]->synchronize();
    this->E[1]->synchronize();
    this->E[2]->synchronize();
}



template <typename _data_type>
template <bool _print>
void decx::conv::_cuda_conv2_async_controller<_data_type>::generate_stream_and_event(de::DH* handle)
{
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        this->S[i] = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
        if (S[i] == NULL) {
            decx::err::CUDA_Stream_access_fail<_print>(handle);
            return;
        }
        this->E[i] = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
        if (E[i] == NULL) {
            decx::err::CUDA_Event_access_fail<_print>(handle);
            return;
        }
    }
}



template <typename _data_type>
template <bool _print, int src_unit_len, int dst_unit_len, int ker_unit_len>
void decx::conv::_cuda_conv2_async_controller<_data_type>::_conv2_MK_alloc(de::DH* handle)
{
    const size_t src_buf_size = static_cast<size_t>(this->_Kparams.src_buf_dims.x) * static_cast<size_t>(this->_Kparams.src_buf_dims.y);
    const size_t dst_buf_size = static_cast<size_t>(this->_Kparams.dst_dims.x) * static_cast<size_t>(this->_Kparams.dst_dims.y);
    const size_t ker_buf_size = static_cast<size_t>(this->_Kparams.ker_dims.x) * static_cast<size_t>(this->_Kparams.ker_dims.y);

#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        if (decx::alloc::_device_malloc(&this->src_buf[i], src_buf_size * src_unit_len, true, this->S[0])) {
            decx::err::device_AllocateFailure<_print>(handle);
            return;
        }
        if (decx::alloc::_device_malloc(&this->dst_buf[i], dst_buf_size * dst_unit_len, true, this->S[0])) {
            decx::err::device_AllocateFailure<_print>(handle);
            return;
        }
        if (decx::alloc::_device_malloc(&this->ker_buf[i], ker_buf_size * ker_unit_len, true, this->S[0])) {
            decx::err::device_AllocateFailure<_print>(handle);
            return;
        }
        // bind pointer to buffer stage
        this->dev_src[i].mem = this->src_buf[i].ptr;
        this->dev_dst[i].mem = this->dst_buf[i].ptr;
    }

    this->dev_src[0]._using = false;
    this->dev_src[1]._using = true;     // in the fake busy state

    this->E[0]->event_record(this->S[0]);
}



template <typename _data_type>
template <uint stream_id, bool _is_MK, typename _kernel_type>
void decx::conv::_cuda_conv2_async_controller<_data_type>::memcpy_src_async_H2D(const uint _mat_id)
{
    decx::conv::_cuConv2_kernel_params* k_params = &this->_Kparams;

    decx::alloc::MIF<void>* dst_ptr = this->dev_src[0]._using ? &this->dev_src[1] : &this->dev_src[0];
    decx::alloc::MIF<void>* _another = this->dev_src[0]._using ? &this->dev_src[0] : &this->dev_src[1];
        
    // shift XY of the pointer
    _data_type* cpy_begin = decx::utils::ptr_shift_xy<void, _data_type>(dst_ptr->mem, this->_Kparams.cpy_shift.x, 
        this->_Kparams.cpy_shift.y, k_params->src_buf_dims.x);

    checkCudaErrors(cudaMemcpy2DAsync(cpy_begin, 
                                        k_params->src_buf_dims.x * sizeof(_data_type), 
                                        k_params->_src_confs._ptr_array[_mat_id],
                                        k_params->_src_confs._pitch * sizeof(_data_type), 
                                        k_params->_src_confs._width * sizeof(_data_type),
                                        k_params->_src_confs._height, 
                                        cudaMemcpyHostToDevice, 
                                        this->S[stream_id]->get_raw_stream_ref()));

    if (_is_MK) {
        void* _kernel_device_ptr = this->dev_src[0]._using ? this->ker_buf[1].ptr : this->ker_buf[0].ptr;

        checkCudaErrors(cudaMemcpy2DAsync(_kernel_device_ptr, 
                                            k_params->ker_dims.x * sizeof(_kernel_type),
                                            k_params->_kernel_confs._ptr_array[_mat_id], 
                                            k_params->_kernel_confs._pitch * sizeof(_kernel_type),
                                            k_params->ker_dims.x * sizeof(_kernel_type), 
                                            k_params->ker_dims.y,
                                            cudaMemcpyHostToDevice, 
                                            this->S[stream_id]->get_raw_stream_ref()));
    }

    decx::utils::set_mutex_memory_state<void, void>(dst_ptr, _another);
    dst_ptr->_using = true;
    _another->_using = false;
    this->E[stream_id]->event_record(this->S[stream_id]);
}



template <typename _data_type>
template <uint stream_id, typename _kernel_type>
void decx::conv::_cuda_conv2_async_controller<_data_type>::memcpy_kernel_async_H2D(const uint _mat_id)
{
    decx::conv::_cuConv2_kernel_params* k_params = &this->_Kparams;

    checkCudaErrors(cudaMemcpy2DAsync(this->ker_buf[0].ptr, 
                                        k_params->ker_dims.x * sizeof(_kernel_type),
                                        (_kernel_type*)this->_Kparams._kernel_confs._ptr, 
                                        k_params->_kernel_confs._pitch * sizeof(_kernel_type),
                                        k_params->ker_dims.x * sizeof(_kernel_type), 
                                        k_params->ker_dims.y,
                                        cudaMemcpyHostToDevice, 
                                        this->S[stream_id]->get_raw_stream_ref()));

    this->E[stream_id]->event_record(this->S[stream_id]);
}



template <typename _data_type>
template <uint stream_id, typename _dst_type>
void decx::conv::_cuda_conv2_async_controller<_data_type>::memcpy_dst_async_D2H(const uint _mat_id)
{
    decx::conv::_cuConv2_kernel_params* k_params = &this->_Kparams;

    void* src_ptr = this->dev_dst[0].leading ? this->dev_dst[0].mem : this->dev_dst[1].mem;

    checkCudaErrors(cudaMemcpyAsync(k_params->_dst_confs._ptr_array[_mat_id],
        src_ptr, k_params->dst_dims.x * k_params->dst_dims.y * sizeof(_dst_type), cudaMemcpyDeviceToHost,
        this->S[stream_id]->get_raw_stream_ref()));

    this->E[stream_id]->event_record(this->S[stream_id]);
}



template <typename _data_type>
template <bool _is_MK>
void decx::conv::_cuda_conv2_async_controller<_data_type>::get_kernel_read_write_ptr(void** read_ptr, void** write_ptr, void** kernel_ptr)
{
    if (this->dev_src[0].leading) {
        *read_ptr = this->dev_src[0].mem;
        *write_ptr = this->dev_dst[0].mem;
        *kernel_ptr = this->ker_buf[0].ptr;
        decx::utils::set_mutex_memory_state<void, void>(&this->dev_dst[0], &this->dev_dst[1]);
    }
    else {
        *read_ptr = this->dev_src[1].mem;
        *write_ptr = this->dev_dst[1].mem;
        *kernel_ptr = _is_MK ? this->ker_buf[1].ptr : this->ker_buf[0].ptr;
        decx::utils::set_mutex_memory_state<void, void>(&this->dev_dst[1], &this->dev_dst[0]);
    }
}



// Should be called after the synchronization
template <typename _data_type>
void decx::conv::_cuda_conv2_async_controller<_data_type>::after_kernel()
{
    if (this->dev_dst[0].leading) {
        this->dev_src[0]._using = false;
    }
    else {
        this->dev_src[1]._using = false;
    }
}


template <typename _data_type>
void decx::conv::_cuda_conv2_async_controller<_data_type>::release()
{
#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        decx::alloc::_device_dealloc(&this->src_buf[i]);
        decx::alloc::_device_dealloc(&this->ker_buf[i]);
        decx::alloc::_device_dealloc(&this->dst_buf[i]);
    }

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        this->S[i]->detach();
        this->E[i]->detach();
    }
}


#endif
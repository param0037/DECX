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


#include "DP2D_1way.cuh"


// --------------------------------------------- DP2D-1way -----------------------------------------------


template <typename _type_in>
decx::blas::cuda_DP2D_configs<_type_in>::cuda_DP2D_configs()
{
    this->_load_byte_changed = false;
    this->_post_proc_conf = NULL;
}

template decx::blas::cuda_DP2D_configs<float>::cuda_DP2D_configs();
template decx::blas::cuda_DP2D_configs<de::Half>::cuda_DP2D_configs();



template <typename _type_in>
template <bool _is_reduce_h>
void decx::blas::cuda_DP2D_configs<_type_in>::generate_config(const uint2 proc_dims, decx::cuda_stream* S, const uint32_t _fp16_accu)
{
    this->_from_dev = false;

    this->_proc_dims = proc_dims;

    uint8_t _proc_align = 1;
    if (sizeof(_type_in) == 4) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 1) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_1B_;
    }
    else if (sizeof(_type_in) == 2) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 8) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_8B_;
    }

    const uint64_t _proc_W_v = decx::utils::ceil<uint64_t>(this->_proc_dims.x, _proc_align);
    this->_dev_mat_dims = make_uint2(_proc_W_v * _proc_align, this->_proc_dims.y);

    this->_first_kernel_config =
        dim3(decx::utils::ceil<uint32_t>(_proc_W_v, _REDUCE2D_BLOCK_DIM_X_), decx::utils::ceil<uint32_t>(this->_proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_));

    const uint32_t _proc_dim_len = _is_reduce_h ? this->_first_kernel_config.x : this->_first_kernel_config.y;

    if (_proc_dim_len > 1)
    {
        const uint2 _postproc_dims = _is_reduce_h ? make_uint2(this->_first_kernel_config.x, this->_proc_dims.y) : 
                                                    make_uint2(this->_proc_dims.x, this->_first_kernel_config.y);
        if (std::is_same<_type_in, de::Half>::value) 
        {
            if (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
                this->_post_proc_conf = new decx::reduce::cuda_reduce2D_1way_configs<float>;
                ((decx::reduce::cuda_reduce2D_1way_configs<float>*)this->_post_proc_conf)->generate_configs<_is_reduce_h>(_postproc_dims, S, false);
            }
            else {
                this->_post_proc_conf = new decx::reduce::cuda_reduce2D_1way_configs<de::Half>;
                ((decx::reduce::cuda_reduce2D_1way_configs<de::Half>*)this->_post_proc_conf)->generate_configs<_is_reduce_h>(_postproc_dims, S, true);
            }
        }
        else if (std::is_same<_type_in, float>::value) {
            this->_post_proc_conf = new decx::reduce::cuda_reduce2D_1way_configs<float>;
            ((decx::reduce::cuda_reduce2D_1way_configs<float>*)this->_post_proc_conf)->generate_configs<_is_reduce_h>(_postproc_dims, S, true);
        }
        else if (std::is_same<_type_in, double>::value) {
            this->_post_proc_conf = new decx::reduce::cuda_reduce2D_1way_configs<double>;
            ((decx::reduce::cuda_reduce2D_1way_configs<double>*)this->_post_proc_conf)->generate_configs<_is_reduce_h>(_postproc_dims, S, true);
        }

        this->_post_proc_needed = true;
    }
    else {
        /*uint32_t _alloc_dst_size = 0;
        if (std::is_same<_type_in, de::Half>::value && _fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            _alloc_dst_size = decx::utils::ceil<uint32_t>(
                _is_reduce_h ? this->_proc_dims.y : this->_proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_ * sizeof(float);
        }
        else {
            _alloc_dst_size = decx::utils::ceil<uint32_t>(
                _is_reduce_h ? this->_proc_dims.y : this->_proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_ * sizeof(_type_in);
        }
        if (decx::alloc::_device_malloc(&this->_dev_dst, _alloc_dst_size, true, S)) {
            Print_Error_Message(4, DEV_ALLOC_FAIL);
            return;
        }*/
        this->_post_proc_needed = false;
    }
}

template void decx::blas::cuda_DP2D_configs<float>::generate_config<true>(const uint2, decx::cuda_stream*, const uint32_t);
template void decx::blas::cuda_DP2D_configs<de::Half>::generate_config<true>(const uint2, decx::cuda_stream*, const uint32_t);

template void decx::blas::cuda_DP2D_configs<float>::generate_config<false>(const uint2, decx::cuda_stream*, const uint32_t);
template void decx::blas::cuda_DP2D_configs<de::Half>::generate_config<false>(const uint2, decx::cuda_stream*, const uint32_t);


template <typename _type_in>
template <bool _is_reduce_h>
void decx::blas::cuda_DP2D_configs<_type_in>::alloc_buffers(decx::cuda_stream* S, const uint32_t _fp16_accu)
{
    if (decx::alloc::_device_malloc(&this->_dev_A, this->_dev_mat_dims.x * this->_dev_mat_dims.y * sizeof(_type_in), true, S) ||
        decx::alloc::_device_malloc(&this->_dev_B, this->_dev_mat_dims.x * this->_dev_mat_dims.y * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Allocation for dst
    if (!this->_post_proc_needed) {
        uint32_t _alloc_dst_size = 0;
        if (std::is_same<_type_in, de::Half>::value && _fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            _alloc_dst_size = decx::utils::align<uint32_t>(
                _is_reduce_h ? this->_proc_dims.y : this->_proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * sizeof(float);
        }
        else {
            _alloc_dst_size = decx::utils::align<uint32_t>(
                _is_reduce_h ? this->_proc_dims.y : this->_proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * sizeof(_type_in);
        }
        if (decx::alloc::_device_malloc(&this->_dev_dst, _alloc_dst_size, true, S)) {
            Print_Error_Message(4, DEV_ALLOC_FAIL);
            return;
        }
    }
}

template void decx::blas::cuda_DP2D_configs<float>::alloc_buffers<true>(decx::cuda_stream*, const uint32_t);
template void decx::blas::cuda_DP2D_configs<float>::alloc_buffers<false>(decx::cuda_stream*, const uint32_t);
template void decx::blas::cuda_DP2D_configs<de::Half>::alloc_buffers<true>(decx::cuda_stream*, const uint32_t);
template void decx::blas::cuda_DP2D_configs<de::Half>::alloc_buffers<false>(decx::cuda_stream*, const uint32_t);


template <typename _type_in>
void decx::blas::cuda_DP2D_configs<_type_in>::release_buffer()
{
    if (!this->_from_dev) {
        decx::alloc::_device_dealloc(&this->_dev_A);
        decx::alloc::_device_dealloc(&this->_dev_B);

        if (!this->postproc_needed()) {
            decx::alloc::_device_dealloc(&this->_dev_dst);
            ((decx::reduce::cuda_reduce2D_1way_configs<_type_in>*)this->_post_proc_conf)->release_buffer();
        }
    }
}

template void decx::blas::cuda_DP2D_configs<float>::release_buffer();
template void decx::blas::cuda_DP2D_configs<de::Half>::release_buffer();


template <typename _type_in>
template <typename _config_type>
decx::reduce::cuda_reduce2D_1way_configs<_config_type>* decx::blas::cuda_DP2D_configs<_type_in>::get_configs_ptr()
{
    return (decx::reduce::cuda_reduce2D_1way_configs<_config_type>*)this->_post_proc_conf;
}

template decx::reduce::cuda_reduce2D_1way_configs<float>* decx::blas::cuda_DP2D_configs<float>::get_configs_ptr();
template decx::reduce::cuda_reduce2D_1way_configs<de::Half>* decx::blas::cuda_DP2D_configs<de::Half>::get_configs_ptr();
template decx::reduce::cuda_reduce2D_1way_configs<float>* decx::blas::cuda_DP2D_configs<de::Half>::get_configs_ptr();


template <typename _type_in>
uint2 decx::blas::cuda_DP2D_configs<_type_in>::get_actual_proc_dims() const
{
    return this->_proc_dims;
}

template uint2 decx::blas::cuda_DP2D_configs<float>::get_actual_proc_dims() const;
template uint2 decx::blas::cuda_DP2D_configs<de::Half>::get_actual_proc_dims() const;



template <typename _type_in>
dim3 decx::blas::cuda_DP2D_configs<_type_in>::get_1st_kernel_config() const
{
    return this->_first_kernel_config;
}

template dim3 decx::blas::cuda_DP2D_configs<float>::get_1st_kernel_config() const;
template dim3 decx::blas::cuda_DP2D_configs<de::Half>::get_1st_kernel_config() const;



template <typename _type_in>
bool decx::blas::cuda_DP2D_configs<_type_in>::postproc_needed() const
{
    return this->_post_proc_needed;
}

template bool decx::blas::cuda_DP2D_configs<float>::postproc_needed() const;
template bool decx::blas::cuda_DP2D_configs<de::Half>::postproc_needed() const;
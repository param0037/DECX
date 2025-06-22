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


#include "DP1D_config.cuh"


#define MODULE_TAG "blas::cuda"


template <typename _type_in>
decx::blas::cuda_DP1D_configs<_type_in>::cuda_DP1D_configs()
{
    this->_load_byte_changed = false;
    this->_post_proc_conf = NULL;
}

template decx::blas::cuda_DP1D_configs<float>::cuda_DP1D_configs();
template decx::blas::cuda_DP1D_configs<de::Half>::cuda_DP1D_configs();
template decx::blas::cuda_DP1D_configs<double>::cuda_DP1D_configs();



template <typename _type_in>
decx::blas::cuda_DP1D_configs<_type_in>::cuda_DP1D_configs(const uint64_t _proc_len, decx::cuda_stream* S,
    const uint32_t _fp16_accu)
{
    int32_t rval = 0;

    this->_from_dev = false;

    this->_proc_len_v1 = _proc_len;

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

    const uint64_t _proc_len_v = decx::utils::idiv_ceil<uint64_t>(_proc_len, _proc_align);
    rval |= this->_dev_A.Allocate(_proc_len_v * _proc_align * sizeof(_type_in), CUDA_DEVICE, de::GetLastError(), true, S);
    rval |= this->_dev_B.Allocate(_proc_len_v * _proc_align * sizeof(_type_in), CUDA_DEVICE, de::GetLastError(), true, S);

    const uint64_t grid_len_k1 = decx::utils::idiv_ceil<uint64_t>(_proc_len_v, _REDUCE1D_BLOCK_DIM_);
    this->_grid_len_k1 = grid_len_k1;

    uint8_t ele_size_dst = 1;

    if (grid_len_k1 > 1) {
        if (std::is_same<_type_in, de::Half>::value && _fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            this->_post_proc_conf = new decx::reduce::cuda_reduce1D_configs<float>;
            ((decx::reduce::cuda_reduce1D_configs<float>*)this->_post_proc_conf)->generate_configs(grid_len_k1, S);
            ele_size_dst = sizeof(float);
        }
        else {
            this->_post_proc_conf = new decx::reduce::cuda_reduce1D_configs<_type_in>;
            ((decx::reduce::cuda_reduce1D_configs<_type_in>*)this->_post_proc_conf)->generate_configs(grid_len_k1, S);
            ele_size_dst = sizeof(_type_in);
        }

        this->_post_proc_needed = true;
    }
    else {
        rval |= this->_dev_dst.Allocate(1 * ele_size_dst, CUDA_DEVICE, de::GetLastError());
        this->_post_proc_needed = false;
    }
}

template decx::blas::cuda_DP1D_configs<float>::cuda_DP1D_configs(const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu);
template decx::blas::cuda_DP1D_configs<de::Half>::cuda_DP1D_configs(const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu);
template decx::blas::cuda_DP1D_configs<double>::cuda_DP1D_configs(const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu);


template <typename _type_in>
bool decx::blas::cuda_DP1D_configs<_type_in>::postproc_needed() const
{
    return this->_post_proc_needed;
}

template bool decx::blas::cuda_DP1D_configs<float>::postproc_needed() const;
template bool decx::blas::cuda_DP1D_configs<de::Half>::postproc_needed() const;
template bool decx::blas::cuda_DP1D_configs<double>::postproc_needed() const;



template <typename _type_in>
uint64_t decx::blas::cuda_DP1D_configs<_type_in>::get_actual_proc_len() const
{
    return this->_proc_len_v1;
}

template uint64_t decx::blas::cuda_DP1D_configs<float>::get_actual_proc_len() const;
template uint64_t decx::blas::cuda_DP1D_configs<de::Half>::get_actual_proc_len() const;
template uint64_t decx::blas::cuda_DP1D_configs<double>::get_actual_proc_len() const;



template <typename _type_in>
uint64_t decx::blas::cuda_DP1D_configs<_type_in>::get_grid_len_k1() const
{
    return this->_grid_len_k1;
}

template uint64_t decx::blas::cuda_DP1D_configs<float>::get_grid_len_k1() const;
template uint64_t decx::blas::cuda_DP1D_configs<de::Half>::get_grid_len_k1() const;
template uint64_t decx::blas::cuda_DP1D_configs<double>::get_grid_len_k1() const;



template <typename _type_in>
template <typename _config_type>
decx::reduce::cuda_reduce1D_configs<_config_type>* decx::blas::cuda_DP1D_configs<_type_in>::get_configs_ptr()
{
    return ((decx::reduce::cuda_reduce1D_configs<_config_type>*)this->_post_proc_conf);
}

template decx::reduce::cuda_reduce1D_configs<float>* decx::blas::cuda_DP1D_configs<float>::get_configs_ptr<float>();
template decx::reduce::cuda_reduce1D_configs<float>* decx::blas::cuda_DP1D_configs<de::Half>::get_configs_ptr<float>();
template decx::reduce::cuda_reduce1D_configs<de::Half>* decx::blas::cuda_DP1D_configs<de::Half>::get_configs_ptr<de::Half>();
template decx::reduce::cuda_reduce1D_configs<double>* decx::blas::cuda_DP1D_configs<double>::get_configs_ptr<double>();



template <typename _type_in>
decx::blas::cuda_DP1D_configs<_type_in>::cuda_DP1D_configs(decx::PtrInfo<void> dev_A, decx::PtrInfo<void> dev_B, 
    const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu)
{
    this->_from_dev = false;

    this->_proc_len_v1 = _proc_len;

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

    const uint64_t _proc_len_v = decx::utils::idiv_ceil<uint64_t>(_proc_len, _proc_align);
    this->_dev_A = dev_A;
    this->_dev_B = dev_B;

    const uint64_t grid_len_k1 = decx::utils::idiv_ceil<uint64_t>(_proc_len_v, _REDUCE1D_BLOCK_DIM_);
    this->_grid_len_k1 = grid_len_k1;

    uint8_t ele_size_dst = 1;

    if (grid_len_k1 > 1) {
        if (std::is_same<_type_in, de::Half>::value && _fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            this->_post_proc_conf = new decx::reduce::cuda_reduce1D_configs<float>;
            ((decx::reduce::cuda_reduce1D_configs<float>*)this->_post_proc_conf)->generate_configs(grid_len_k1, S);
            ele_size_dst = sizeof(float);
        }
        else {
            this->_post_proc_conf = new decx::reduce::cuda_reduce1D_configs<_type_in>;
            ((decx::reduce::cuda_reduce1D_configs<_type_in>*)this->_post_proc_conf)->generate_configs(grid_len_k1, S);
            ele_size_dst = sizeof(_type_in);
        }

        this->_post_proc_needed = true;
    }
    else {
        this->_dev_dst.Allocate(1 * ele_size_dst, CUDA_DEVICE, de::GetLastError(), true, S);
        this->_post_proc_needed = false;
    }
}

template decx::blas::cuda_DP1D_configs<float>::cuda_DP1D_configs(decx::PtrInfo<void> dev_A, decx::PtrInfo<void> dev_B, 
    const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu);
template decx::blas::cuda_DP1D_configs<de::Half>::cuda_DP1D_configs(decx::PtrInfo<void> dev_A, decx::PtrInfo<void> dev_B, 
    const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu);
template decx::blas::cuda_DP1D_configs<double>::cuda_DP1D_configs(decx::PtrInfo<void> dev_A, decx::PtrInfo<void> dev_B, 
    const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu);




template <typename _type_in>
void decx::blas::cuda_DP1D_configs<_type_in>::relase_buffer()
{
    if (this->_from_dev) {
        this->_dev_A.Free();
        this->_dev_B.Free();
    }
    if (!this->_post_proc_needed) {
        this->_dev_dst.Free();
    }
}

template void decx::blas::cuda_DP1D_configs<float>::relase_buffer();
template void decx::blas::cuda_DP1D_configs<de::Half>::relase_buffer();
template void decx::blas::cuda_DP1D_configs<double>::relase_buffer();


template <typename _type_in>
decx::blas::cuda_DP1D_configs<_type_in>::~cuda_DP1D_configs()
{
    this->relase_buffer();
}

template decx::blas::cuda_DP1D_configs<float>::~cuda_DP1D_configs();
template decx::blas::cuda_DP1D_configs<de::Half>::~cuda_DP1D_configs();
template decx::blas::cuda_DP1D_configs<double>::~cuda_DP1D_configs();

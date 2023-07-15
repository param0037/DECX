/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "DMA_callers.cuh"
#include "../../classes/classes_util.h"


template <bool _print>
void decx::bp::_DMA_memcpy1D(const void* src, void* dst, const size_t cpy_size,
    cudaMemcpyKind flag, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamDefault);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    checkCudaErrors(cudaMemcpyAsync(dst, src, cpy_size,
        flag, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}


template void decx::bp::_DMA_memcpy1D<true>(const void* src, void* dst, const size_t cpy_size,
    cudaMemcpyKind flag, de::DH* handle);

template void decx::bp::_DMA_memcpy1D<false>(const void* src, void* dst, const size_t cpy_size,
    cudaMemcpyKind flag, de::DH* handle);


template <bool _print>
void decx::bp::_DMA_memcpy2D(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
    const size_t cpy_width, const size_t height, cudaMemcpyKind flag, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaStreamNonBlocking);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst, pitchdst, src, pitchsrc, cpy_width,
        height, flag, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}


template void decx::bp::_DMA_memcpy2D<true>(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
    const size_t cpy_width, const size_t height, cudaMemcpyKind flag, de::DH* handle);


template void decx::bp::_DMA_memcpy2D<false>(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
    const size_t cpy_width, const size_t height, cudaMemcpyKind flag, de::DH* handle);



template <bool _print>
void decx::bp::_DMA_memcpy3D(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
    const size_t _plane_size_src, const size_t _plane_size_dst, const size_t cpy_width, const size_t height, const size_t times, cudaMemcpyKind flag, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaStreamNonBlocking);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    for (int i = 0; i < times; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync((uint8_t*)dst + i * _plane_size_src, pitchdst, 
            (uint8_t*)src + i * _plane_size_dst, pitchsrc,
            cpy_width, height, 
            flag, S->get_raw_stream_ref()));
    }
    
    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}

template void decx::bp::_DMA_memcpy3D<true>(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
    const size_t _plane_size_src, const size_t _plane_size_dst, const size_t cpy_width, const size_t height, const size_t times, cudaMemcpyKind flag, de::DH* handle);


template void decx::bp::_DMA_memcpy3D<false>(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
    const size_t _plane_size_src, const size_t _plane_size_dst, const size_t cpy_width, const size_t height, const size_t times, cudaMemcpyKind flag, de::DH* handle);



decx::bp::memcpy2D_multi_dims_optimizer::memcpy2D_multi_dims_optimizer(const void* raw_src, const decx::_matrix_layout& src_layout,
    void* dst, const decx::_matrix_layout& dst_layout)
{
    this->_src_layout = src_layout;
    this->_dst_layout = dst_layout;

    this->_raw_src = raw_src;
    this->_dst = dst;
}




void decx::bp::memcpy2D_multi_dims_optimizer::memcpy2D_optimizer(const uint2 actual_dims_src, const uint2 start, const uint2 cpy_sizes)
{
    const uint32_t& __size = this->_src_layout._single_element_size;

    if (start.x == 0 && cpy_sizes.x == this->_src_layout.width) {
        this->_opt_cpy_type = cpy_dim_types::CPY_1D;
        this->_start_src = (uint8_t*)this->_raw_src + start.y * this->_src_layout.pitch * __size;
        this->_cpy_sizes.x = this->_src_layout.pitch * cpy_sizes.y * __size;
    }
    else {
        this->_opt_cpy_type = cpy_dim_types::CPY_2D;
        this->_start_src = DECX_PTR_SHF_XY<const void, const uint8_t>(this->_raw_src, start.y,
            start.x * __size, this->_src_layout.pitch * __size);

        this->_cpy_sizes.x = cpy_sizes.x * __size;
        this->_cpy_sizes.y = cpy_sizes.y;
    }
}




template <bool _print>
void decx::bp::memcpy2D_multi_dims_optimizer::execute_DMA(const cudaMemcpyKind memcpykind, de::DH* handle)
{
    const uint32_t& __size = this->_src_layout._single_element_size;

    switch (this->_opt_cpy_type)
    {
    case decx::bp::memcpy2D_multi_dims_optimizer::cpy_dim_types::CPY_1D:
        decx::bp::_DMA_memcpy1D<_print>(this->_start_src, this->_dst, this->_cpy_sizes.x,
            memcpykind, handle);
        break;

    case decx::bp::memcpy2D_multi_dims_optimizer::cpy_dim_types::CPY_2D:
        decx::bp::_DMA_memcpy2D<_print>(this->_start_src, this->_dst, this->_src_layout.pitch * __size,
            this->_dst_layout.pitch * __size, this->_cpy_sizes.x, this->_cpy_sizes.y,
            memcpykind, handle);
        break;

    default:
        break;
    }
}


template void decx::bp::memcpy2D_multi_dims_optimizer::execute_DMA<true>(const cudaMemcpyKind memcpykind, de::DH* handle);
template void decx::bp::memcpy2D_multi_dims_optimizer::execute_DMA<false>(const cudaMemcpyKind memcpykind, de::DH* handle);



decx::bp::memcpy3D_multi_dims_optimizer::memcpy3D_multi_dims_optimizer(const decx::_tensor_layout& _src, const void* src_ptr,
    const decx::_tensor_layout& _dst, void* dst_ptr)
{
    this->_src_layout = _src;
    this->_dst_layout = _dst;

    this->_raw_src = src_ptr;
    this->_dst = dst_ptr;
}



void decx::bp::memcpy3D_multi_dims_optimizer::memcpy3D_optimizer(const uint3 actual_dims_src, const uint3 start, const uint3 cpy_sizes)
{
    if ((start.x == 0 && cpy_sizes.x == actual_dims_src.x) &&
        (start.y == 0 && cpy_sizes.y == actual_dims_src.y)) 
    {
        this->_opt_cpy_type = cpy_dim_types::CPY_1D;
        this->_start_src = (uint8_t*)this->_raw_src + start.z * this->_src_layout.dp_x_wp * this->_src_layout._single_element_size;

        this->_cpy_sizes.x = this->_src_layout.dp_x_wp * this->_src_layout._single_element_size * cpy_sizes.z;
    }
    else {
        if (start.x == 0 && cpy_sizes.x == actual_dims_src.x) 
        {
            this->_opt_cpy_type = cpy_dim_types::CPY_2D;

            this->_start_src = DECX_PTR_SHF_XY<const void, const uint8_t>(this->_raw_src,
                start.z, 
                this->_src_layout.dpitch * start.y * this->_src_layout._single_element_size,
                this->_src_layout.dp_x_wp * this->_src_layout._single_element_size);

            this->_cpy_sizes.x = this->_src_layout.dpitch * cpy_sizes.y * this->_src_layout._single_element_size;
            this->_cpy_sizes.y = cpy_sizes.z;
        }
        else {
            this->_opt_cpy_type = cpy_dim_types::CPY_3D;

            this->_start_src = DECX_PTR_SHF_XY<const void, const uint8_t>(this->_raw_src,
                start.z, 
                (this->_src_layout.dpitch * start.y + start.x) * this->_src_layout._single_element_size,
                this->_src_layout.dp_x_wp * this->_src_layout._single_element_size);

            this->_cpy_sizes.x = cpy_sizes.x;
            this->_cpy_sizes.y = cpy_sizes.y;
            this->_cpy_sizes.z = cpy_sizes.z;
        }
    }
}




template <bool _print>
void decx::bp::memcpy3D_multi_dims_optimizer::execute_DMA(const cudaMemcpyKind memcpykind, de::DH* handle)
{
    const uint32_t& _size_ = this->_src_layout._single_element_size;

    switch (this->_opt_cpy_type)
    {
    case decx::bp::memcpy2D_multi_dims_optimizer::cpy_dim_types::CPY_1D:
        decx::bp::_DMA_memcpy1D<_print>(this->_start_src, this->_dst, this->_cpy_sizes.x * this->_cpy_sizes.y * _size_,
            memcpykind, handle);
        break;

    case decx::bp::memcpy2D_multi_dims_optimizer::cpy_dim_types::CPY_2D:
        decx::bp::_DMA_memcpy2D<_print>(this->_start_src, this->_dst, this->_src_layout.dp_x_wp * _size_,
            this->_dst_layout.dp_x_wp * _size_, this->_cpy_sizes.x * _size_, this->_cpy_sizes.y,
            memcpykind, handle);
        break;

    case decx::bp::memcpy2D_multi_dims_optimizer::cpy_dim_types::CPY_3D:
        decx::bp::_DMA_memcpy3D<_print>(this->_start_src, this->_dst, this->_src_layout.dpitch * _size_,
            this->_dst_layout.dpitch * _size_, this->_src_layout.dp_x_wp * _size_, this->_dst_layout.dp_x_wp * _size_,
            this->_cpy_sizes.x, this->_cpy_sizes.y, this->_cpy_sizes.z, memcpykind, handle);
        break;

    default:
        break;
    }
}

template void decx::bp::memcpy3D_multi_dims_optimizer::execute_DMA<true>(const cudaMemcpyKind memcpykind, de::DH* handle);
template void decx::bp::memcpy3D_multi_dims_optimizer::execute_DMA<false>(const cudaMemcpyKind memcpykind, de::DH* handle);
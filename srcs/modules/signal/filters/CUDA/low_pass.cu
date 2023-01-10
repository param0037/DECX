/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "low_pass.cuh"


__global__ void
decx::signal::GPUK::cu_ideal_LP1D_cpl32(const float4* __restrict src, 
                                        float4* __restrict dst, 
                                        const size_t _proc_len, 
                                        const size_t real_bound,
                                        const size_t cutoff_freq)
{
    size_t dex = threadIdx.x + blockIdx.x * blockDim.x;

    bool _is_eff = false;
    de::CPf recv[2], store[2];

    if (dex < _proc_len) {
        *((float4*)recv) = src[dex];
        _is_eff = ((dex * 2) < cutoff_freq) || ((dex * 2) > (real_bound - cutoff_freq - 1));
        store[0] = _is_eff ? recv[0] : de::CPf(0, 0);
        
        _is_eff = ((dex * 2 + 1) < cutoff_freq) || ((dex * 2 + 1) > (real_bound - cutoff_freq - 1));
        store[1] = _is_eff ? recv[1] : de::CPf(0, 0);
        dst[dex] = *((float4*)store);
    }
}



__global__ void
decx::signal::GPUK::cu_ideal_LP2D_cpl32(const float4* __restrict src, 
                                        float4* __restrict dst, 
                                        const uint2 _proc_dims,         // in float4 (vec2 of datatype of de::CPf)
                                        const uint2 real_bound, 
                                        const uint2 cutoff_freq,
                                        const uint pitch)               // in float4 (vec2 of datatype of de::CPf)
{
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex = 0;

    bool _is_effy_axis = false, _is_effx_axis = false;
    de::CPf recv[2], store[2];

    if (idx < _proc_dims.y && idy < _proc_dims.x) {
        dex = idx * pitch + idy;
        *((float4*)recv) = src[dex];
        _is_effy_axis = (idx < cutoff_freq.y) || (idx > (real_bound.y - cutoff_freq.y - 1));
        _is_effx_axis = ((idy * 2) < cutoff_freq.x) || ((idy * 2) > (real_bound.x - cutoff_freq.x - 1));

        store[0] = (_is_effy_axis && _is_effx_axis) ? recv[0] : de::CPf(0, 0);

        _is_effx_axis = ((idy * 2 + 1) < cutoff_freq.x) || ((idy * 2 + 1) > (real_bound.x - cutoff_freq.x - 1));
        store[1] = (_is_effy_axis && _is_effx_axis) ? recv[1] : de::CPf(0, 0);

        dst[dex] = *((float4*)store);
    }
}



_DECX_API_ de::DH 
de::signal::cuda::LowPass1D_Ideal(de::GPU_Vector& src, de::GPU_Vector& dst, const size_t cutoff_frequency)
{
    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    const size_t max_freq = _src->length / 2;
    if (cutoff_frequency > max_freq) {
        Print_Error_Message(4, INVALID_PARAM);
        decx::err::InvalidParam(&handle);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }
    
    const size_t _proc_len_v2 = _src->_length / 2;
    decx::signal::GPUK::cu_ideal_LP1D_cpl32 << <decx::utils::ceil<size_t>(_proc_len_v2, decx::cuP.prop.maxThreadsPerBlock),
        decx::cuP.prop.maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)_src->Vec.ptr, (float4*)_dst->Vec.ptr, _proc_len_v2, _src->length, cutoff_frequency);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH
de::signal::cuda::LowPass2D_Ideal(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D cutoff_frequency)
{
    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    const uint2 max_freq = make_uint2(_src->width / 2, _src->height / 2);
    if (cutoff_frequency.x > max_freq.x || cutoff_frequency.y > max_freq.y) {
        Print_Error_Message(4, INVALID_PARAM);
        decx::err::InvalidParam(&handle);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    const dim3 grid(decx::utils::ceil<uint>(_src->height, 16),
                    decx::utils::ceil<uint>(_src->pitch / 2, 16));
    const dim3 gpu_thread(16, 16);
    decx::signal::GPUK::cu_ideal_LP2D_cpl32 << <grid, gpu_thread, 0, S->get_raw_stream_ref() >> > (
            (float4*)_src->Mat.ptr, 
            (float4*)_dst->Mat.ptr, 
            make_uint2(_src->pitch / 2, _src->height), 
            make_uint2(_src->width, _src->height), 
            make_uint2(cutoff_frequency.x, cutoff_frequency.y), 
            _src->pitch / 2);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    decx::err::Success(&handle);
    return handle;
}
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "windows.cuh"
#include "../../../core/cudaStream_management/cudaEvent_package.h"
#include "../../../core/cudaStream_management/cudaStream_package.h"


__global__ void
decx::signal::GPUK::cu_Gaussian_Window1D_cpl32(const float4* __restrict         src,
                                               float4* __restrict               dst, 
                                               const float                      u, 
                                               const float                      sigma, 
                                               const size_t                     _proc_len, 
                                               const size_t                     real_bound)
{
    size_t dex = threadIdx.x + blockIdx.x * blockDim.x;
    float4 recv, store;
    long long axis_value = 0, half_len = real_bound / 2;
    bool _half_pass = false;
    float g_weight = 0, _2_x_sigma_sq = __fmul_rn(-2.f, sigma * sigma);

    if (dex < _proc_len) {
        recv = src[dex];

        _half_pass = dex * 2 > half_len;
        axis_value = dex * 2 - (long long)_half_pass * (real_bound - 1);
        g_weight = __fsub_rn(axis_value, u);
        g_weight = __expf(__fdividef(__fmul_rn(g_weight, g_weight), _2_x_sigma_sq));

        store.x = __fmul_rn(recv.x, g_weight);
        store.y = __fmul_rn(recv.y, g_weight);

        _half_pass = dex * 2 + 1 > half_len;
        axis_value = dex * 2 + 1 - (long long)_half_pass * (real_bound - 1);
        g_weight = __fsub_rn(__ll2float_rn(axis_value), u);
        g_weight = __expf(__fdividef(__fmul_rn(g_weight, g_weight), _2_x_sigma_sq));

        store.z = __fmul_rn(recv.z, g_weight);
        store.w = __fmul_rn(recv.w, g_weight);

        dst[dex] = store;
    }
}



__global__ void
decx::signal::GPUK::cu_Triangluar_Window1D_cpl32(const float4* __restrict       src,
                                                float4* __restrict              dst, 
                                                const long long                 origin, 
                                                const size_t                    radius,
                                                const size_t                    _proc_len,
                                                const size_t                    real_bound)
{
    size_t dex = threadIdx.x + blockIdx.x * blockDim.x;
    float4 recv, store;
    long long axis_value = 0, half_len = real_bound / 2, dist = 0;
    bool _half_pass = false, _is_excced = false;
    float g_weight = 0;

    if (dex < _proc_len) {
        recv = src[dex];
        
        _half_pass = dex * 2 > half_len;
        axis_value = dex * 2 - (long long)_half_pass * (real_bound - 1);
        dist = llabs(axis_value - origin);
        _is_excced = dist > radius;
        g_weight = __ll2float_rn(dist);
        g_weight = __fsub_rn(1.f, __fdividef(g_weight, __ll2float_rn(radius)));
        store.x = _is_excced ? 0.f : __fmul_rn(recv.x, g_weight);
        store.y = _is_excced ? 0.f : __fmul_rn(recv.y, g_weight);

        _half_pass = dex * 2 + 1 > half_len;
        axis_value = dex * 2 + 1 - (long long)_half_pass * (real_bound - 1);
        dist = llabs(axis_value - origin);
        _is_excced = dist > radius;
        g_weight = __ll2float_rn(dist);
        g_weight = __fsub_rn(1.f, __fdividef(g_weight, __ll2float_rn(radius)));
        store.z = _is_excced ? 0.f : __fmul_rn(recv.z, g_weight);
        store.w = _is_excced ? 0.f : __fmul_rn(recv.w, g_weight);

        dst[dex] = store;
    }
}



__global__ void
decx::signal::GPUK::cu_Cone_Window2D_cpl32(const float4* __restrict         src, 
                                           float4* __restrict               dst, 
                                           const uint2                      origin, 
                                           const float                       radius, 
                                           const uint2                      _proc_dims,
                                           const uint2                      real_bound, 
                                           const uint                       pitch)
{
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex = 0;

    int Xaxis_value = 0, Yaxis_value = 0,
        half_lenX = real_bound.x / 2, half_lenY = real_bound.y / 2;

    bool _is_hfps_Xaxis = false, _is_hfps_Yaxis = false, in_range = false;
    float g_weight = 0,
        x_component = 0, y_component = 0,
        dist = 0;

    float4 recv, store;

    if (idx < _proc_dims.y&& idy < _proc_dims.x) {
        dex = idx * pitch + idy;
        recv = src[dex];

        _is_hfps_Xaxis = idy * 2 > half_lenX;                                   _is_hfps_Yaxis = idx > half_lenY;
        Xaxis_value = idy * 2 - (int)_is_hfps_Xaxis * (real_bound.x - 1);       Yaxis_value = idx - (int)_is_hfps_Yaxis * (real_bound.y - 1);
        Xaxis_value = Xaxis_value - origin.x;                                   Yaxis_value = Yaxis_value - origin.y;
        dist = sqrtf(__int2float_rn(Xaxis_value * Xaxis_value + Yaxis_value * Yaxis_value));
        in_range = dist < radius;
        g_weight = __fsub_rn(1.f, __fdividef(dist, radius));
        store.x = in_range ? __fmul_rn(recv.x, g_weight) : 0;
        store.y = in_range ? __fmul_rn(recv.y, g_weight) : 0;

        _is_hfps_Xaxis = idy * 2 + 1 > half_lenX;                                   _is_hfps_Yaxis = idx > half_lenY;
        Xaxis_value = idy * 2 + 1 - (int)_is_hfps_Xaxis * (real_bound.x - 1);       Yaxis_value = idx - (int)_is_hfps_Yaxis * (real_bound.y - 1);
        Xaxis_value = Xaxis_value - origin.x;                                   Yaxis_value = Yaxis_value - origin.y;
        dist = sqrtf(__int2float_rn(Xaxis_value * Xaxis_value + Yaxis_value * Yaxis_value));
        in_range = dist < radius;
        g_weight = __fsub_rn(1.f, __fdividef(dist, radius));
        store.z = in_range ? __fmul_rn(recv.z, g_weight) : 0;
        store.w = in_range ? __fmul_rn(recv.w, g_weight) : 0;

        dst[dex] = store;
    }
}



__global__ void
decx::signal::GPUK::cu_Gaussian_Window2D_cpl32_no_correlation(const float4* __restrict src,
                                                              float4* __restrict dst,
                                                              const float2 u,
                                                              const float2 sigma,
                                                              const uint2 _proc_dims,         // in float4 (vec2 of datatype of de::CPf)
                                                              const uint2 real_bound, 
                                                              const uint pitch)               // in float4 (vec2 of datatype of de::CPf)
{
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex = 0;

    int Xaxis_value = 0, Yaxis_value = 0, 
        half_lenX = real_bound.x / 2, half_lenY = real_bound.y / 2;

    bool _is_hfps_Xaxis = false, _is_hfps_Yaxis = false;
    float g_weight = 0, 
          x_component = 0, y_component = 0;

    float4 recv, store;

    if (idx < _proc_dims.y && idy < _proc_dims.x) {
        dex = idx * pitch + idy;
        recv = src[dex];
        
        _is_hfps_Xaxis = idy * 2 > half_lenX;                                   _is_hfps_Yaxis = idx > half_lenY;
        Xaxis_value = idy * 2 - (int)_is_hfps_Xaxis * (real_bound.x - 1);       Yaxis_value = idx - (int)_is_hfps_Yaxis * (real_bound.y - 1);
        x_component = __fsub_rn(Xaxis_value, u.x);                              y_component = __fsub_rn(Yaxis_value, u.y);
        x_component = __fmul_rn(x_component, x_component);                      y_component = __fmul_rn(y_component, y_component);
        x_component = __fdividef(x_component, __fmul_rn(sigma.x, sigma.x));     y_component = __fdividef(y_component, __fmul_rn(sigma.y, sigma.y));
        g_weight = __expf(__fdividef(__fadd_rn(x_component, y_component), -2.f));

        store.x = __fmul_rn(recv.x, g_weight);
        store.y = __fmul_rn(recv.y, g_weight);

        _is_hfps_Xaxis = idy * 2 + 1 > half_lenX;                               _is_hfps_Yaxis = idx > half_lenY;
        Xaxis_value = idy * 2 + 1 - (int)_is_hfps_Xaxis * (real_bound.x - 1);   Yaxis_value = idx - (int)_is_hfps_Yaxis * (real_bound.y - 1);
        x_component = __fsub_rn(Xaxis_value, u.x);                              y_component = __fsub_rn(Yaxis_value, u.y);
        x_component = __fmul_rn(x_component, x_component);                      y_component = __fmul_rn(y_component, y_component);
        x_component = __fdividef(x_component, __fmul_rn(sigma.x, sigma.x));     y_component = __fdividef(y_component, __fmul_rn(sigma.y, sigma.y));
        g_weight = __expf(__fdividef(__fadd_rn(x_component, y_component), -2.f));

        store.z = __fmul_rn(recv.z, g_weight);
        store.w = __fmul_rn(recv.w, g_weight);

        dst[dex] = store;
    }
}



__global__ void
decx::signal::GPUK::cu_Gaussian_Window2D_cpl32(const float4* __restrict src,
                                               float4* __restrict dst,
                                               const float2 u,
                                               const float2 sigma,
                                               const uint2 _proc_dims,         // in float4 (vec2 of datatype of de::CPf)
                                               const uint2 real_bound, 
                                               const float p,
                                               const uint pitch)               // in float4 (vec2 of datatype of de::CPf)
{
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t dex = 0;

    int Xaxis_value = 0, Yaxis_value = 0, 
        half_lenX = real_bound.x / 2, half_lenY = real_bound.y / 2;

    bool _is_hfps_Xaxis = false, _is_hfps_Yaxis = false;
    float g_weight = 0, 
          x_component = 0, y_component = 0, xy_component = 0;

    float4 recv, store;

    if (idx < _proc_dims.y && idy < _proc_dims.x) {
        dex = idx * pitch + idy;
        recv = src[dex];
        
        _is_hfps_Xaxis = idy * 2 > half_lenX;                                       _is_hfps_Yaxis = idx > half_lenY;
        Xaxis_value = idy * 2 - (int)_is_hfps_Xaxis * (real_bound.x - 1);           Yaxis_value = idx - (int)_is_hfps_Yaxis * (real_bound.y - 1);
        x_component = __fsub_rn(Xaxis_value, u.x);                                  y_component = __fsub_rn(Yaxis_value, u.y);
        xy_component = __fdividef(__fmul_rn(2 * p, __fmul_rn(x_component, y_component)), __fmul_rn(sigma.x, sigma.y));
        x_component = __fmul_rn(x_component, x_component);                          y_component = __fmul_rn(y_component, y_component);
        x_component = __fdividef(x_component, __fmul_rn(sigma.x, sigma.x));         y_component = __fdividef(y_component, __fmul_rn(sigma.y, sigma.y));
        g_weight = __fsub_rn(__fadd_rn(x_component, y_component), xy_component);
        g_weight = __expf(__fdividef(g_weight, __fmul_rn(-2.f, 1.f - p * p)));

        store.x = __fmul_rn(recv.x, g_weight);
        store.y = __fmul_rn(recv.y, g_weight);

        _is_hfps_Xaxis = idy * 2 + 1 > half_lenX;                                   _is_hfps_Yaxis = idx > half_lenY;
        Xaxis_value = idy * 2 + 1 - (int)_is_hfps_Xaxis * (real_bound.x - 1);       Yaxis_value = idx - (int)_is_hfps_Yaxis * (real_bound.y - 1);
        x_component = __fsub_rn(Xaxis_value, u.x);                                  y_component = __fsub_rn(Yaxis_value, u.y);
        xy_component = __fdividef(__fmul_rn(2 * p, __fmul_rn(x_component, y_component)), __fmul_rn(sigma.x, sigma.y));
        x_component = __fmul_rn(x_component, x_component);                          y_component = __fmul_rn(y_component, y_component);
        x_component = __fdividef(x_component, __fmul_rn(sigma.x, sigma.x));         y_component = __fdividef(y_component, __fmul_rn(sigma.y, sigma.y));
        g_weight = __fsub_rn(__fadd_rn(x_component, y_component), xy_component);
        g_weight = __expf(__fdividef(g_weight, __fmul_rn(-2.f, 1.f - p * p)));

        store.z = __fmul_rn(recv.z, g_weight);
        store.w = __fmul_rn(recv.w, g_weight);

        dst[dex] = store;
    }
}



_DECX_API_ de::DH
de::signal::cuda::Gaussian_Window1D(de::GPU_Vector& src, de::GPU_Vector& dst, const float u, const float sigma)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    const size_t _proc_len_v2 = _src->_length / 2;
    decx::signal::GPUK::cu_Gaussian_Window1D_cpl32 << <decx::utils::ceil<size_t>(_proc_len_v2, decx::cuda::_get_cuda_prop().maxThreadsPerBlock),
        decx::cuda::_get_cuda_prop().maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)_src->Vec.ptr, (float4*)_dst->Vec.ptr, u, sigma, _proc_len_v2, _src->length);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH
de::signal::cuda::Triangular_Window1D(de::GPU_Vector& src, de::GPU_Vector& dst, const long long origin, const size_t radius)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    const size_t _proc_len_v2 = _src->_length / 2;
    decx::signal::GPUK::cu_Triangluar_Window1D_cpl32 << <decx::utils::ceil<size_t>(_proc_len_v2, decx::cuda::_get_cuda_prop().maxThreadsPerBlock),
        decx::cuda::_get_cuda_prop().maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)_src->Vec.ptr, (float4*)_dst->Vec.ptr, origin, radius, _proc_len_v2, _src->length);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH 
de::signal::cuda::Cone_Window2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D origin, const float radius)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    const dim3 grid(decx::utils::ceil<uint>(_src->Height(), 16),
        decx::utils::ceil<uint>(_src->Pitch() / 2, 16));
    const dim3 gpu_thread(16, 16);
    decx::signal::GPUK::cu_Cone_Window2D_cpl32 << <grid, gpu_thread, 0, S->get_raw_stream_ref() >> > (
        (float4*)_src->Mat.ptr,
        (float4*)_dst->Mat.ptr,
        make_uint2(origin.x, origin.y),
        radius,
        make_uint2(_src->Pitch() / 2, _src->Height()),
        make_uint2(_src->Width(), _src->Height()),
        _src->Pitch() / 2);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH
de::signal::cuda::Gaussian_Window2D(
    de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    const dim3 grid(decx::utils::ceil<uint>(_src->Height(), 16),
        decx::utils::ceil<uint>(_src->Pitch() / 2, 16));
    const dim3 gpu_thread(16, 16);
    if (p == 0) {
        decx::signal::GPUK::cu_Gaussian_Window2D_cpl32_no_correlation << <grid, gpu_thread, 0, S->get_raw_stream_ref() >> > (
            (float4*)_src->Mat.ptr,
            (float4*)_dst->Mat.ptr,
            make_float2(u.x, u.y),
            make_float2(sigma.x, sigma.y),
            make_uint2(_src->Pitch() / 2, _src->Height()),
            make_uint2(_src->Width(), _src->Height()),
            _src->Pitch() / 2);
    }
    else {
        if (!(p < 1.f && p > -1.f)) {
            decx::err::InvalidParam(&handle);
            Print_Error_Message(4, INVALID_PARAM);
            return handle;
        }
        decx::signal::GPUK::cu_Gaussian_Window2D_cpl32 << <grid, gpu_thread, 0, S->get_raw_stream_ref() >> > (
            (float4*)_src->Mat.ptr,
            (float4*)_dst->Mat.ptr,
            make_float2(u.x, u.y),
            make_float2(sigma.x, sigma.y),
            make_uint2(_src->Pitch() / 2, _src->Height()),
            make_uint2(_src->Width(), _src->Height()),
            p,
            _src->Pitch() / 2);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    decx::err::Success(&handle);
    return handle;
}
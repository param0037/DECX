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


#ifndef _MATARRAY_CHANNEL_SUM_H_
#define _MATARRAY_CHANNEL_SUM_H_

#include "../../core/basic.h"
#include "../../classes/MatrixArray.h"
#include "../../classes/GPU_MatrixArray.h"
#include "../../classes/classes_util.h"

#ifndef GNU_CPUcodes
#define fragment_4 2048 * 2048
#endif


namespace de
{
#ifdef _DECX_CUDA_CODES_
    namespace cuda
    {
        template <typename T>
        de::DH MatArray_Merge_sum(de::MatrixArray<T>& src, de::Matrix<T>& dst);

        template <typename T>
        de::DH MatArray_Merge_sum(de::GPU_MatrixArray<T>& src, de::GPU_Matrix<T>& dst);
#endif
    }
}


using decx::_MatrixArray;
using decx::_GPU_MatrixArray;


namespace decx
{
#ifdef _DECX_CUDA_CODES_
    static void MatArray_channel_sum_4_f(_MatrixArray<float>* src, _Matrix<float>* dst, de::DH* handle);

    static void dev_MatArray_channel_sum_4_f(_GPU_MatrixArray<float>* src, _GPU_Matrix<float>* dst, de::DH* handle);

#else

#endif
}


#ifdef _DECX_CUDA_CODES_
__global__
/**
* @param _plane : frag_plane in float4
* @param Wfrag : in float4, Wfrag / 4
*/
void cu_MatArr_channel_sum4(float4* src, float4* dst, const size_t _plane,
    const int Hfrag, const int Wfrag, const int Mat_num)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 reg_tmp, ans = make_float4(0.f, 0.f, 0.f, 0.f);
    size_t dex = tidx * Wfrag + tidy;

    if (tidx < Hfrag && tidy < Wfrag) {
        for (int i = 0; i < Mat_num; ++i) {
            reg_tmp = src[dex + i * _plane];

            ans.x = __fadd_rn(reg_tmp.x, ans.x);
            ans.y = __fadd_rn(reg_tmp.y, ans.y);
            ans.z = __fadd_rn(reg_tmp.z, ans.z);
            ans.w = __fadd_rn(reg_tmp.w, ans.w);
        }
        dst[dex] = ans;
    }
}


#define _MatArr_CpyTo2D_vec4(plane_shift, type_, vec_type_)    \
{    \
if (dev_src1._using) {    \
    for (int k = 0; k < Mat_num; ++k) {    \
        checkCudaErrors(cudaMemcpy2DAsync(dev_src2.mem + k * frag_plane, Wsrc * sizeof(vec_type_),    \
            (vec_type_*)src->MatptrArr.ptr[k] + plane_shift, src->pitch * sizeof(type_), Wsrc * sizeof(vec_type_), H_cpy_src,    \
            cudaMemcpyHostToDevice, S_cpyTo));    \
    }    \
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_src2, &dev_src1);    \
}    \
else {    \
    for (int k = 0; k < Mat_num; ++k) {    \
        checkCudaErrors(cudaMemcpy2DAsync(dev_src1.mem + k * frag_plane, Wsrc * sizeof(vec_type_),    \
            (vec_type_*)src->MatptrArr.ptr[k] + plane_shift, src->pitch * sizeof(type_), Wsrc * sizeof(vec_type_), H_cpy_src,    \
            cudaMemcpyHostToDevice, S_cpyTo));    \
    }    \
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_src1, &dev_src2);    \
}    \
}


#define _MatArr_CpyBack2D_vec4(plane_shift, type_, vec_type_)    \
{    \
if (dev_dst1.leading) {    \
    checkCudaErrors(cudaMemcpy2DAsync((vec_type_*)dst->Mat.ptr + plane_shift, dst->pitch * sizeof(type_),    \
        dev_dst1.mem, Wsrc * sizeof(vec_type_), Wsrc * sizeof(vec_type_), H_cpy_dst,    \
        cudaMemcpyDeviceToHost, S_cpyBack));    \
}    \
else {    \
    checkCudaErrors(cudaMemcpy2DAsync((vec_type_*)dst->Mat.ptr + plane_shift, dst->pitch * sizeof(type_),    \
        dev_dst2.mem, Wsrc * sizeof(vec_type_), Wsrc * sizeof(vec_type_), H_cpy_dst,    \
        cudaMemcpyDeviceToHost, S_cpyBack));    \
}    \
}


__global__
/**
* Regard the whole device memory block as a 1D vector
* @param len : _plane in float4
* @param Wfrag : in float4, Wfrag / 4
*/
void cu_MatArr_channel_sum4_1D(float4* src, float4* dst, const size_t len, const int Mat_num)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 _ans = make_float4(0, 0, 0, 0), tmp;

    if (tid < len) {
        for (int i = 0; i < Mat_num; ++i) {
            tmp = src[tid + (size_t)i * len];
            _ans.x = __fadd_rn(tmp.x, _ans.x);
            _ans.y = __fadd_rn(tmp.y, _ans.y);
            _ans.z = __fadd_rn(tmp.z, _ans.z);
            _ans.w = __fadd_rn(tmp.w, _ans.w);
        }
        dst[tid] = _ans;
    }
}


#endif




static
void decx::MatArray_channel_sum_4_f(_MatrixArray<float>* src, _Matrix<float>* dst, de::DH* handle)
{
    const int Wsrc = decx::utils::ceil<int>(src->width, 4)/* * 4*/;        // align to 4
    const int Hsrc = src->height;
    const int Mat_num = src->ArrayNumber;

    const int Hfrag = (int)(decx::utils::ceil<size_t>(fragment_4 / 4, Mat_num * Wsrc));
    const size_t frag_plane = Hfrag * Wsrc;
    const size_t true_frag = frag_plane * Mat_num;

    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * (true_frag + frag_plane) * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    
    decx::alloc::MIF<float4> dev_src1, dev_src2,
        dev_dst1, dev_dst2;
    dev_src1.mem = dev_tmp.ptr;
    dev_dst1.mem = dev_tmp.ptr + true_frag;
    dev_src2.mem = dev_tmp.ptr + true_frag + frag_plane;
    dev_dst2.mem = dev_tmp.ptr + true_frag * 2 + frag_plane;

    cudaStream_t S_ker, S_cpyTo, S_cpyBack;
    checkCudaErrors(cudaStreamCreate(&S_ker));
    checkCudaErrors(cudaStreamCreate(&S_cpyTo));
    checkCudaErrors(cudaStreamCreate(&S_cpyBack));

    int H_cpy_src = Hsrc > Hfrag ? Hfrag : Hsrc,
        H_cpy_dst;

    for (int k = 0; k < Mat_num; ++k) {
        checkCudaErrors(cudaMemcpy2DAsync(dev_src1.mem + k * frag_plane, Wsrc * sizeof(float4), 
            src->MatptrArr.ptr[k], src->pitch * sizeof(float), Wsrc * sizeof(float4), H_cpy_src, 
            cudaMemcpyHostToDevice, S_cpyTo));
    }
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_src1, &dev_src2);    

    checkCudaErrors(cudaDeviceSynchronize());
    
    const int __iter = decx::utils::ceil<int>(Hsrc, Hfrag);
    size_t plane_shift_src = frag_plane,
        plane_shift_dst = 0;

    const dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);
    const dim3 grid(decx::utils::ceil<int>(Hfrag, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(Wsrc, _BLOCK_DEFAULT_));

    for (int i = 0; i < __iter; ++i) {
        H_cpy_dst = (Hsrc - i * Hfrag) > 0 ? Hfrag : (Hsrc - i * Hfrag);
        H_cpy_src = (Hsrc - (i + 1) * Hfrag) > Hfrag ? Hfrag : (Hsrc - (i + 1) * Hfrag);

        if (i > 0) {
            _MatArr_CpyBack2D_vec4(plane_shift_dst, float, float4);        // judge leading
            plane_shift_dst += frag_plane;
        }
        if (dev_src1.leading) {
            cu_MatArr_channel_sum4 << <grid, block, 0, S_ker >> > (
                dev_src1.mem, dev_dst1.mem, frag_plane, Hfrag, Wsrc, Mat_num);

            decx::utils::set_mutex_memory_state<float4, float4>(&dev_dst1, &dev_dst2);
            dev_src1._using = true;
        }
        else {
            cu_MatArr_channel_sum4 << <grid, block, 0, S_ker >> > (
                dev_src2.mem, dev_dst2.mem, frag_plane, Hfrag, Wsrc, Mat_num);

            decx::utils::set_mutex_memory_state<float4, float4>(&dev_dst2, &dev_dst1);
            dev_src2._using = true;
        }

        if (i < __iter - 1) {
            _MatArr_CpyTo2D_vec4(plane_shift_src, float, float4);        // judge _using
        }
        plane_shift_src += frag_plane;
        
        checkCudaErrors(cudaDeviceSynchronize());
        dev_src1._using = false;
        dev_src2._using = false;
    }

    H_cpy_dst = (Hsrc - __iter * Hfrag) > 0 ? Hfrag : (Hsrc - (__iter - 1) * Hfrag);
    _MatArr_CpyBack2D_vec4(plane_shift_dst, float, float4);        // judge leading
    
    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S_ker));
    checkCudaErrors(cudaStreamDestroy(S_cpyTo));
    checkCudaErrors(cudaStreamDestroy(S_cpyBack));
}


#ifndef GNU_CPUcodes
static void decx::dev_MatArray_channel_sum_4_f(_GPU_MatrixArray<float>* src, _GPU_Matrix<float>* dst, de::DH* handle)
{
    if (src->width != dst->width || src->height != dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::MDim_Not_Matching(handle);
        return;
    }
    
    const size_t _len_eq = src->_plane / 4;
    
    cu_MatArr_channel_sum4_1D << <decx::utils::ceil<size_t>(_len_eq, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
        reinterpret_cast<float4*>(src->MatArr.ptr), reinterpret_cast<float4*>(dst->Mat.ptr), _len_eq, src->ArrayNumber);
}
#endif



template <typename T>
de::DH de::cuda::MatArray_Merge_sum(de::MatrixArray<T>& src, de::Matrix<T>& dst)
{
    _MatrixArray<T>* _src = dynamic_cast<_MatrixArray<T>*>(&src);
    _Matrix<T>* _dst = dynamic_cast<_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    if (!cuP.is_init) {
        decx::Not_init(&handle);
    }

    if (_src->width != _dst->width || _src->height != _dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    decx::MatArray_channel_sum_4_f(_src, _dst, &handle);

    return handle;
}

template _DECX_API_ de::DH de::cuda::MatArray_Merge_sum(de::MatrixArray<float>& src, de::Matrix<float>& dst);



#ifndef GNU_CPUcodes
template <typename T>
de::DH de::cuda::MatArray_Merge_sum(de::GPU_MatrixArray<T>& src, de::GPU_Matrix<T>& dst)
{
    _GPU_MatrixArray<T>* _src = dynamic_cast<_GPU_MatrixArray<T>*>(&src);
    _GPU_Matrix<T>* _dst = dynamic_cast<_GPU_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    if (!cuP.is_init) {
        decx::Not_init(&handle);
    }

    if (_src->width != _dst->width || _src->height != _dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    decx::dev_MatArray_channel_sum_4_f(_src, _dst, &handle);

    return handle;
}

template _DECX_API_ de::DH de::cuda::MatArray_Merge_sum(de::GPU_MatrixArray<float>& src, de::GPU_Matrix<float>& dst);


#endif

#ifndef GNU_CPUcodes
#ifdef fragment_4
#undef fragment_4
#endif
#endif


#endif
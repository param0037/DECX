/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CONV2_LARGE_KERNEL_H_
#define _CONV2_LARGE_KERNEL_H_


#include "../../../classes/classes_util.h"
#include "../../../classes/GPU_Matrix.h"
#include "conv2_large_kernel.cuh"


using decx::_GPU_Matrix;
using decx::alloc::MIF;


namespace de
{
    namespace cuda
    {
        /* 
        * This function is especially designed for the backward propagation of convolution layer in CNN, Thus, the way of padding is a bit 
        * different from that in a traditional convolution. The padding size is according to the dimensions of kernel. 
        */
        _DECX_API_ de::DH Conv2_large_kernel(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& kernel, de::GPU_Matrix<float>& dst, const int flag);
    }
}


namespace decx
{
    /*
    * This function operates a dot product of an sub-matrix and kernel. ATTENTION: dev_tmp1 is first used.
    */
    static void sub_matrix_dot_kernel_fp32(const size_t total_len, MIF<float4> *dev_tmp1, MIF<float4> *dev_tmp2, float* dst_ptr,
        float4* kernel_ptr, cudaStream_t *S_ker, const size_t dst_dex);


    static void dev_sConv2_LK_border_ignore(_GPU_Matrix<float>* src, _GPU_Matrix<float>* kernel, _GPU_Matrix<float>* dst);

    // DIM(src) = DIM(kernel)
    static void dev_sConv2_LK_border_zero(_GPU_Matrix<float>* src, _GPU_Matrix<float>* kernel, _GPU_Matrix<float>* dst);
}



static void decx::sub_matrix_dot_kernel_fp32(const size_t total_len, MIF<float4>* dev_tmp1, MIF<float4>* dev_tmp2, float* dst_ptr,
    float4* kernel_ptr, cudaStream_t* S_ker, const size_t dst_dex)
{
    size_t grid = total_len / 2, thr_num = total_len / 2;
    cudaMemsetAsync(dev_tmp2->mem, 0, total_len * sizeof(float4), *S_ker);

    if (total_len > REDUCTION_BLOCK_SIZE)
    {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);
        
        cu_sConv2_LK_start << <grid, REDUCTION_BLOCK_SIZE, 0, *S_ker >> > (
            dev_tmp1->mem, kernel_ptr, dev_tmp2->mem, thr_num);
        decx::utils::set_mutex_memory_state<float4, float4>(dev_tmp2, dev_tmp1);

        thr_num = grid / 2;

        while (1) {
            if (decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE) == 1) { break; }
            grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);

            if (dev_tmp1->leading) {
                cu_sConv2_LK << <grid, REDUCTION_BLOCK_SIZE, 0, *S_ker >> > (dev_tmp1->mem, dev_tmp2->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_sConv2_LK << <grid, REDUCTION_BLOCK_SIZE, 0, *S_ker >> > (dev_tmp2->mem, dev_tmp1->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_tmp1, dev_tmp2);
            }
            thr_num = grid / 2;
        }
        thr_num = grid / 2;
        if (dev_tmp1->leading) {
            cu_sConv2_LK_end << <1, REDUCTION_BLOCK_SIZE, 0, *S_ker >> > (dev_tmp1->mem, dst_ptr, thr_num, dst_dex);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_tmp2, dev_tmp1);
        }
        else {
            cu_sConv2_LK_end << <1, REDUCTION_BLOCK_SIZE, 0, *S_ker >> > (dev_tmp2->mem, dst_ptr, thr_num, dst_dex);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_tmp1, dev_tmp2);
        }
    }
}



void decx::dev_sConv2_LK_border_ignore(_GPU_Matrix<float>* src, _GPU_Matrix<float>* kernel, _GPU_Matrix<float>* dst)
{
    const uint Wsrc = src->width;
    const uint Hsrc = src->height;

    cudaStream_t S_ker, S_cpy;
    checkCudaErrors(cudaStreamCreate(&S_ker));
    checkCudaErrors(cudaStreamCreate(&S_cpy));

    size_t total_len = decx::utils::ceil<size_t>(((size_t)kernel->pitch * (size_t)kernel->height), 8) * 2;
    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 3 * total_len * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    // reset the memory zone
    cudaMemset(dev_tmp.ptr, 0, 3 * total_len * sizeof(float4));

    decx::alloc::MIF<float4> dev_tmp1, dev_tmp2, dev_tmp3;
    dev_tmp1.mem = dev_tmp.ptr;                                // buffer_1, acts as the sub-matrix loaded input
    dev_tmp2.mem = dev_tmp.ptr + total_len;                    // buffer_2, acts as the temp buffer
    dev_tmp3.mem = dev_tmp.ptr + (total_len << 1);            // buffer_2, acts as the sub-matrix loaded input

    // copy the sub-matrix of src to the buffer
    checkCudaErrors(cudaMemcpy2DAsync(dev_tmp1.mem, kernel->pitch * sizeof(float), 
        src->Mat.ptr, src->pitch * sizeof(float), kernel->width * sizeof(float), kernel->height, 
        cudaMemcpyDeviceToDevice, S_cpy));
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_tmp1, &dev_tmp2);
    dev_tmp3._using = false;
    checkCudaErrors(cudaDeviceSynchronize());

    const size_t ker_len = (size_t)dst->width * (size_t)dst->height;
    int k_x = 0, k_y = 0, k_x_cpy = 0, k_y_cpy = 0;
    for (size_t i = 0; i < ker_len; ++i) 
    {
        k_x = i / dst->width;
        k_y = i % dst->width;
        if (i < ker_len - 1) {
            k_x_cpy = (i + 1) / dst->width;
            k_y_cpy = (i + 1) % dst->width;
            if (!dev_tmp3._using) {        // dev_tmp1 is just loaded with the data from source matrix
                checkCudaErrors(cudaMemcpy2DAsync(dev_tmp3.mem, kernel->pitch * sizeof(float),
                    src->Mat.ptr + (size_t)k_x_cpy * (size_t)src->pitch + (size_t)k_y_cpy, src->pitch * sizeof(float),
                    kernel->width * sizeof(float), kernel->height, cudaMemcpyDeviceToDevice, S_cpy));
                dev_tmp3._using = true;
            }
            else {
                checkCudaErrors(cudaMemcpy2DAsync(dev_tmp1.mem, kernel->pitch * sizeof(float),
                    src->Mat.ptr + (size_t)k_x_cpy * (size_t)src->pitch + (size_t)k_y_cpy, src->pitch * sizeof(float),
                    kernel->width * sizeof(float), kernel->height, cudaMemcpyDeviceToDevice, S_cpy));
                dev_tmp3._using = false;
            }
        }
        else {
            // dev_tmp1 is just loaded with the data from source matrix
            if (!dev_tmp3._using) {     dev_tmp3._using = true; }
            else { dev_tmp3._using = false; }
        }
        if (dev_tmp3._using) {
            decx::sub_matrix_dot_kernel_fp32(total_len, &dev_tmp1, &dev_tmp2,
                dst->Mat.ptr, reinterpret_cast<float4*>(kernel->Mat.ptr), &S_ker, k_x * dst->pitch + k_y);
        }
        else {
            decx::sub_matrix_dot_kernel_fp32(total_len, &dev_tmp3, &dev_tmp2,
                dst->Mat.ptr, reinterpret_cast<float4*>(kernel->Mat.ptr), &S_ker, k_x * dst->pitch + k_y);
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S_ker));
    checkCudaErrors(cudaStreamDestroy(S_cpy));
}




void decx::dev_sConv2_LK_border_zero(_GPU_Matrix<float>* src, _GPU_Matrix<float>* kernel, _GPU_Matrix<float>* dst)
{
    const uint Wsrc = src->width;
    const uint Hsrc = src->height;

    cudaStream_t S_ker;
    checkCudaErrors(cudaStreamCreate(&S_ker));

    size_t total_len = decx::utils::ceil<size_t>(((size_t)kernel->pitch * (size_t)kernel->height), 8) * 2;
    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * total_len * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::alloc::MIF<float4> dev_tmp1, dev_tmp2;
    dev_tmp1.mem = dev_tmp.ptr;
    dev_tmp2.mem = dev_tmp.ptr + total_len;

    // reset the memory zone
    cudaMemset(dev_tmp.ptr, 0, 2 * total_len * sizeof(float4));

    int r_x = dst->height / 2, r_y = dst->width / 2;
    // copy the sub-matrix of src to the buffer
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(dev_tmp1.mem) + (size_t)r_x * (size_t)kernel->pitch + (size_t)r_y,
        kernel->pitch * sizeof(float),
        src->Mat.ptr,
        src->pitch * sizeof(float),
        (kernel->width - r_y) * sizeof(float),
        kernel->height - r_x,
        cudaMemcpyDeviceToDevice, S_ker));
    
    const size_t ker_len = (size_t)dst->width * (size_t)dst->height;
    int k_x_cpy = 0, k_y_cpy = 0, k_x_tmp = 0, k_y_tmp = 0, copyW, copyH;

    for (size_t i = 0; i < ker_len; ++i)
    {
        if (i < ker_len - 1) {
            k_x_cpy = decx::utils::clamp_min<int>(((i + 1) / dst->width) - r_x, 0);
            k_y_cpy = decx::utils::clamp_min<int>(((i + 1) % dst->width) - r_y, 0);
            
            k_x_tmp = decx::utils::clamp_min<int>(r_x - ((i + 1) / dst->width), 0);
            k_y_tmp = decx::utils::clamp_min<int>(r_y - ((i + 1) % dst->width), 0);

            copyH = kernel->height - decx::utils::Iabs(((i + 1) / dst->width) - r_x);
            copyW = kernel->width - decx::utils::Iabs(((i + 1) % dst->width) - r_y);

            cudaMemsetAsync(dev_tmp1.mem, 0, total_len * sizeof(float4), S_ker);

            checkCudaErrors(cudaMemcpy2DAsync(
                reinterpret_cast<float*>(dev_tmp1.mem) + (size_t)k_x_tmp * (size_t)kernel->pitch + (size_t)k_y_tmp, kernel->pitch * sizeof(float),
                src->Mat.ptr + (size_t)k_x_cpy * (size_t)src->pitch + (size_t)k_y_cpy, src->pitch * sizeof(float),
                copyW * sizeof(float), copyH,
                cudaMemcpyDeviceToDevice, S_ker));
        }
        
        decx::sub_matrix_dot_kernel_fp32(total_len, &dev_tmp1, &dev_tmp2,
            dst->Mat.ptr, reinterpret_cast<float4*>(kernel->Mat.ptr), &S_ker, (i / dst->width) * dst->pitch + (i % dst->width));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S_ker));
}




de::DH de::cuda::Conv2_large_kernel(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& kernel, de::GPU_Matrix<float>& dst, const int flag)
{
    _GPU_Matrix<float>* _src = dynamic_cast<_GPU_Matrix<float>*>(&src);
    _GPU_Matrix<float>* _kernel = dynamic_cast<_GPU_Matrix<float>*>(&kernel);
    _GPU_Matrix<float>* _dst = dynamic_cast<_GPU_Matrix<float>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_sConv2_LK_border_ignore(_src, _kernel, _dst);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_sConv2_LK_border_zero(_src, _kernel, _dst);
        break;
    default:
        break;
    }

    return handle;
}


#endif
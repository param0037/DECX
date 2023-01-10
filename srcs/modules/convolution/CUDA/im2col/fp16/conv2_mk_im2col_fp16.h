/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV2_MK_IM2COL_FP16_H_
#define _CONV2_MK_IM2COL_FP16_H_

#include "../../../../classes/Tensor.h"
#include "../../../../classes/TensorArray.h"
#include "../im2col.cuh"
#include "eq_GEMM_fp16.cuh"
#include "../../../conv_utils.h"


namespace decx
{
    static void hconv2_BC_r8_mk_im2col(decx::_Tensor* src, decx::_TensorArray* kernel, 
        decx::_Tensor* dst, const int accu_flag, decx::cuda_stream *S);



    static void hconv2_NB_r8_mk_im2col(decx::_Tensor* src, decx::_TensorArray* kernel, 
        decx::_Tensor* dst, const int accu_flag, decx::cuda_stream* S);
}



static void decx::hconv2_BC_r8_mk_im2col(decx::_Tensor* src, decx::_TensorArray* kernel, 
    decx::_Tensor* dst, const int accu_flag, decx::cuda_stream* S)
{

    const int2 kernel_shift = make_int2(8 - kernel->height / 2, 8 - kernel->width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->width, 
                                       src->height,
                                       kernel->tensor_num,
                                       decx::DATA_STORE_TYPE::Page_Locked);

    dst->re_construct(src->type, src->width, src->height, kernel->tensor_num, decx::DATA_STORE_TYPE::Page_Locked);

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const uint2 ker_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->plane[0] * (size_t)kernel->dpitch, 128) * 128, 
                                         decx::utils::ceil<uint>(kernel->tensor_num, 64) * 64);

    const uint actual_dst_buf_W = decx::utils::ceil<uint>(kernel->tensor_num, 8) * 8;

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4, 
                                       dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->width, 64) * 64 + 16) * (size_t)src->dpitch, 
                                            decx::utils::ceil<uint>(src->height, 16) * 16 + 16);
    
    const ulong2 I2C_dims = make_ulong2(kernel->plane[0] * (size_t)kernel->dpitch, 
                                        eq_src_dims.x * eq_src_dims.y);

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->tensor_num, 64) * 64, 
                                         decx::utils::ceil<size_t>(I2C_dims.y, 128) * 128);
    
    decx::PtrInfo<float4> src_buf, dst_buf, I2C_buf, ker_buf;
    
    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dst_buf, actual_dst_buf_W * dst_buf_dim.y * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.x * I2C_dims.y * sizeof(float4) / 8)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->tensor_num; ++i) {
        cudaMemcpy2DAsync(DECX_PTR_SHF_XY<float4, de::Half>(ker_buf.ptr, i, 0, ker_buf_dim.x),      kernel->dpitch * kernel->width * sizeof(de::Half),
                          kernel->TensptrArr.ptr[i],                                                kernel->dp_x_wp * sizeof(de::Half),
                          kernel->dpitch * kernel->width * sizeof(de::Half),                        kernel->height,
                          cudaMemcpyHostToDevice,                                                   S->get_raw_stream_ref());
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        DECX_PTR_SHF_XY<float4, de::Half>(src_buf.ptr, 8, 8, src_buf_dim.x),                        src_buf_dim.x * sizeof(de::Half),
        src->Tens.ptr,                                                                              src->dp_x_wp * sizeof(de::Half),
        src->dp_x_wp * sizeof(de::Half),                                                            src->height, 
        cudaMemcpyHostToDevice,                                                                     S->get_raw_stream_ref()));

    int2 thread_bounds = make_int2(eq_src_dims.x / 4, eq_src_dims.y);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.y, 16), decx::utils::ceil<int>(thread_bounds.x, 16));

    decx::conv::GPUK::cu_sIm2Col_r8_within << <grid_I2C, block_I2C, 0, S->get_raw_stream_ref() >> > (src_buf.ptr,
                                                             I2C_buf.ptr,
                                                             kernel_shift,
                                                             thread_bounds,
                                                             src_buf_dim.x / 8,
                                                             I2C_dims.x / 8,
                                                             make_int2(kernel->width, kernel->height),
                                                             kernel->dpitch / 8);

    const ulong2 MatA_load_bounds = make_ulong2(I2C_dims.x / 8, I2C_dims.y / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(dst_buf_dim.y / 128, dst_buf_dim.x / 64);

    switch (accu_flag)
    {
    case decx::conv_property::half_conv_ordinary:
        decx::conv::GPUK::cu_conv_eq_mm_fp16 << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf.ptr, 
                                                        ker_buf.ptr, 
                                                        dst_buf.ptr, 
                                                        MatA_load_bounds,
                                                        actual_dst_buf_W / 8,
                                                        ker_buf_dim.x / 8,
                                                        ker_buf_dim.x / 128);
        break;
    case decx::conv_property::half_conv_accurate:
        decx::conv::GPUK::cu_conv_eq_mm_fp16_accu << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf.ptr, 
                                                        ker_buf.ptr, 
                                                        dst_buf.ptr, 
                                                        MatA_load_bounds,
                                                        actual_dst_buf_W / 8,
                                                        ker_buf_dim.x / 8,
                                                        ker_buf_dim.x / 128);
        break;
    default:
        break;
    }
    
    // copy back the data
    checkCudaErrors(cudaMemcpyAsync(
        dst->Tens.ptr, dst_buf.ptr, dst->total_bytes, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&dst_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}



static void decx::hconv2_NB_r8_mk_im2col(decx::_Tensor* src, decx::_TensorArray* kernel, 
    decx::_Tensor* dst, const int accu_flag, decx::cuda_stream* S)
{
    const int2 kernel_shift = make_int2(8 - kernel->height / 2, 8 - kernel->width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->width - (kernel->width / 2) * 2, 
                                       src->height - (kernel->height / 2) * 2,
                                       kernel->tensor_num,
                                       decx::DATA_STORE_TYPE::Page_Locked);

    dst->re_construct(src->type,
        src->width - (kernel->width / 2) * 2,
        src->height - (kernel->height / 2) * 2,
        kernel->tensor_num,
        decx::DATA_STORE_TYPE::Page_Locked);

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const uint2 ker_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->plane[0] * (size_t)kernel->dpitch, 128) * 128, 
                                         decx::utils::ceil<uint>(kernel->tensor_num, 64) * 64);

    const uint actual_dst_buf_W = decx::utils::ceil<uint>(kernel->tensor_num, 8) * 8;

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4, 
                                       dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->width, 64) * 64 + kernel_shift.y * 2) * (size_t)src->dpitch, 
                                            decx::utils::ceil<uint>(src->height, 16) * 16 + kernel_shift.x * 2);
    
    const ulong2 I2C_dims = make_ulong2(kernel->plane[0] * (size_t)kernel->dpitch, 
                                        eq_src_dims.x * eq_src_dims.y);

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->tensor_num, 64) * 64, 
                                         decx::utils::ceil<size_t>(I2C_dims.y, 128) * 128);
    
    decx::PtrInfo<float4> src_buf, dst_buf, I2C_buf, ker_buf;
    
    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dst_buf, actual_dst_buf_W * dst_buf_dim.y * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.x * I2C_dims.y * sizeof(float4) / 8, true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->tensor_num; ++i) {
        checkCudaErrors(
            cudaMemcpy2DAsync(reinterpret_cast<de::Half*>(ker_buf.ptr) + i * ker_buf_dim.x,             kernel->dpitch * kernel->width * sizeof(de::Half),
                              kernel->TensptrArr.ptr[i],                                                kernel->dp_x_wp * sizeof(de::Half),
                              kernel->dpitch * kernel->width * sizeof(de::Half),                        kernel->height,
                              cudaMemcpyHostToDevice,                                                   S->get_raw_stream_ref()));
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(src_buf.ptr) + kernel_shift.x * src_buf_dim.x + kernel_shift.y * src->dpitch,             src_buf_dim.x * sizeof(de::Half), 
        src->Tens.ptr,                                                                              src->dp_x_wp * sizeof(de::Half),
        src->dp_x_wp * sizeof(de::Half),                                                            src->height, 
        cudaMemcpyHostToDevice,                                                                     S->get_raw_stream_ref()));

    int2 thread_bounds = make_int2(eq_src_dims.x / 4, eq_src_dims.y);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.y, 16), decx::utils::ceil<int>(thread_bounds.x, 16));

    decx::conv::GPUK::cu_sIm2Col_r8_within << <grid_I2C, block_I2C, 0, S->get_raw_stream_ref() >> > (src_buf.ptr,
                                                             I2C_buf.ptr,
                                                             kernel_shift,
                                                             thread_bounds,
                                                             src_buf_dim.x / 8,
                                                             I2C_dims.x / 8,
                                                             make_int2(kernel->width, kernel->height),
                                                             kernel->dpitch / 8);

    const ulong2 MatA_load_bounds = make_ulong2(I2C_dims.x / 8, I2C_dims.y / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(dst_buf_dim.y / 128, dst_buf_dim.x / 64);

    switch (accu_flag)
    {
    case decx::conv_property::half_conv_ordinary:
        decx::conv::GPUK::cu_conv_eq_mm_fp16 << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf.ptr, 
                                                        ker_buf.ptr, 
                                                        dst_buf.ptr, 
                                                        MatA_load_bounds,
                                                        actual_dst_buf_W / 8,
                                                        ker_buf_dim.x / 8,
                                                        ker_buf_dim.x / 128);
        break;
    case decx::conv_property::half_conv_accurate:
        decx::conv::GPUK::cu_conv_eq_mm_fp16_accu << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf.ptr, 
                                                        ker_buf.ptr, 
                                                        dst_buf.ptr, 
                                                        MatA_load_bounds,
                                                        actual_dst_buf_W / 8,
                                                        ker_buf_dim.x / 8,
                                                        ker_buf_dim.x / 128);
        break;
    default:
        break;
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Tens.ptr, dst_buf.ptr, dst->total_bytes, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&dst_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}




#endif
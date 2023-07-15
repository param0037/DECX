/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifndef _RELU_H_
#define _RELU_H_

#include "../../core/basic.h"
#include "../../classes/GPU_Matrix.h"
#include "../../classes/GPU_Vector.h"


using decx::_GPU_Matrix;
using decx::_GPU_Vector;


namespace de
{
    namespace nn
    {
        _DECX_API_ de::DH ReLU_f(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst);


        _DECX_API_ de::DH ReLU_f_Rec(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<uchar>& rec_map);


        _DECX_API_ de::DH ReLU_f_Backward(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<uchar>& rec_map);
    }
}


namespace decx
{
    void _dev_ReLU_f_caller_Mat(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, de::DH* handle);


    void _dev_ReLU_f_caller_Mat_Rec(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, _GPU_Matrix<uchar>* map, de::DH* handle);


    void _dev_ReLU_f_caller_Mat_Backward(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, _GPU_Matrix<uchar>* map, de::DH* handle);
}



__global__
/**
* @param len : It is in type float4
*/
void cu_general_ReLU_f(float4* src, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmp;
    int _sign;

    if (tid < len) {
        tmp = src[tid];

        _sign = ~(*((int*)&tmp.x) & 0x80000000) >> 31;
        tmp.x = __fmul_rn(__int2float_rn(_sign), tmp.x);

        _sign = ~(*((int*)&tmp.y) & 0x80000000) >> 31;
        tmp.y = __fmul_rn(__int2float_rn(_sign), tmp.y);

        _sign = ~(*((int*)&tmp.z) & 0x80000000) >> 31;
        tmp.z = __fmul_rn(__int2float_rn(_sign), tmp.z);

        _sign = ~(*((int*)&tmp.w) & 0x80000000) >> 31;
        tmp.w = __fmul_rn(__int2float_rn(_sign), tmp.w);

        dst[tid] = tmp;
    }
}



#define _MaxPooling_REC_(_buffer, _label, _order){                            \
    _sign = (~(*((uint*)&_buffer._label) & 0x80000000)) >> 31;                \
    _buffer._label = __fmul_rn(__int2float_rn(_sign), _buffer._label);        \
    dst_mask |= (_sign << _order);                                            \
}


__global__
/**
* @brief : This kernel function can not only apply 'ReLU' operation, but also can record each
*    element that is greater than zero.
* The record media : de::GPU_Matrix<uchar> -> each bit represents a state of the element in the
* corresponding position. sbit = 0 if is less than zero, otherwise 1.
* 
* @param len : It is in type float4 * 2 one thread process 8 floats
*/
void cu_general_ReLU_f_rec(float4* src, float4* dst, uchar* rec_map, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmp1, tmp2;
    int _sign;
    uchar dst_mask = 0;

    if (tid < len) {
        tmp1 = src[tid * 2];
        tmp2 = src[tid * 2 + 1];

        _MaxPooling_REC_(tmp1, x, 4);
        _MaxPooling_REC_(tmp1, y, 5);
        _MaxPooling_REC_(tmp1, z, 6);
        _MaxPooling_REC_(tmp1, w, 7);

        _MaxPooling_REC_(tmp2, x, 0);
        _MaxPooling_REC_(tmp2, y, 1);
        _MaxPooling_REC_(tmp2, z, 2);
        _MaxPooling_REC_(tmp2, w, 3);
        
        dst[tid * 2] = tmp1;
        dst[tid * 2 + 1] = tmp2;
        rec_map[tid] = dst_mask;
    }
}



__global__
/**
* @param len : It is in type float4 * 2 one thread process 8 floats
*/
void cu_general_ReLU_f_backward(float4* src, float4* dst, uchar* rec_map, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmp1, tmp2;
    int _sign;
    uchar dst_mask = 0;

    if (tid < len) {
        tmp1 = src[tid * 2];
        tmp2 = src[tid * 2 + 1];
        dst_mask = rec_map[tid];
        
        _sign = ((dst_mask & 0x10) >> 4);
        tmp1.x = __fmul_rn(__int2float_rn(_sign), tmp1.x);
        _sign = ((dst_mask & 0x20) >> 5);
        tmp1.y = __fmul_rn(__int2float_rn(_sign), tmp1.y);
        _sign = ((dst_mask & 0x40) >> 6);
        tmp1.z = __fmul_rn(__int2float_rn(_sign), tmp1.z);
        _sign = ((dst_mask & 0x80) >> 7);
        tmp1.w = __fmul_rn(__int2float_rn(_sign), tmp1.w);

        _sign = ((dst_mask & 0x01) >> 0);
        tmp2.x = __fmul_rn(__int2float_rn(_sign), tmp2.x);
        _sign = ((dst_mask & 0x02) >> 1);
        tmp2.y = __fmul_rn(__int2float_rn(_sign), tmp2.y);
        _sign = ((dst_mask & 0x04) >> 2);
        tmp2.z = __fmul_rn(__int2float_rn(_sign), tmp2.z);
        _sign = ((dst_mask & 0x08) >> 3);
        tmp2.w = __fmul_rn(__int2float_rn(_sign), tmp2.w);

        dst[tid * 2] = tmp1;
        dst[tid * 2 + 1] = tmp2;
        
    }
}



void decx::_dev_ReLU_f_caller_Mat(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, de::DH* handle)
{
    if (src->width != dst->width || src->height != dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return;
    }
    const size_t len = static_cast<size_t>(src->pitch) * static_cast<size_t>(src->height);

    cu_general_ReLU_f << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock>> > (
        reinterpret_cast<float4*>(src->Mat.ptr), reinterpret_cast<float4*>(dst->Mat.ptr), len / 4);
}



void decx::_dev_ReLU_f_caller_Mat_Rec(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, _GPU_Matrix<uchar>* map, de::DH* handle)
{
    if (src->width != dst->width || src->height != dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return;
    }
    const size_t len = static_cast<size_t>(src->pitch) * static_cast<size_t>(src->height);

    cu_general_ReLU_f_rec << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock>> > (
        reinterpret_cast<float4*>(src->Mat.ptr), reinterpret_cast<float4*>(dst->Mat.ptr), map->Mat.ptr, len / 8);
}



void decx::_dev_ReLU_f_caller_Mat_Backward(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, _GPU_Matrix<uchar>* map, de::DH* handle)
{
    if (src->width != dst->width || src->height != dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return;
    }
    const size_t len = static_cast<size_t>(src->pitch) * static_cast<size_t>(src->height);

    cu_general_ReLU_f_backward << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock>> > (
        reinterpret_cast<float4*>(src->Mat.ptr), reinterpret_cast<float4*>(dst->Mat.ptr), map->Mat.ptr, len / 8);
}




de::DH de::nn::ReLU_f(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_GPU_Matrix<float>* _src = dynamic_cast<decx::_GPU_Matrix<float>*>(&src);
    decx::_GPU_Matrix<float>* _dst = dynamic_cast<decx::_GPU_Matrix<float>*>(&dst);

    decx::_dev_ReLU_f_caller_Mat(_src, _dst, &handle);

    return handle;
}


de::DH de::nn::ReLU_f_Rec(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<uchar>& rec_map)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_GPU_Matrix<float>* _src = dynamic_cast<decx::_GPU_Matrix<float>*>(&src);
    decx::_GPU_Matrix<float>* _dst = dynamic_cast<decx::_GPU_Matrix<float>*>(&dst);
    decx::_GPU_Matrix<uchar>* _map = dynamic_cast<decx::_GPU_Matrix<uchar>*>(&rec_map);

    decx::_dev_ReLU_f_caller_Mat_Rec(_src, _dst, _map, &handle);

    return handle;
}



de::DH de::nn::ReLU_f_Backward(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<uchar>& rec_map)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_GPU_Matrix<float>* _src = dynamic_cast<decx::_GPU_Matrix<float>*>(&src);
    decx::_GPU_Matrix<float>* _dst = dynamic_cast<decx::_GPU_Matrix<float>*>(&dst);
    decx::_GPU_Matrix<uchar>* _map = dynamic_cast<decx::_GPU_Matrix<uchar>*>(&rec_map);

    decx::_dev_ReLU_f_caller_Mat_Backward(_src, _dst, _map, &handle);

    return handle;
}


#endif
/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include "../../core/basic.h"
#include "../../classes/GPU_Matrix.h"
#include "../../classes/GPU_Vector.h"


using decx::_GPU_Matrix;
using decx::_GPU_Vector;


namespace de
{
    namespace nn
    {
        _DECX_API_ de::DH sigmoid_f(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst);

        _DECX_API_ de::DH sigmoid_f(de::GPU_Vector<float>& src, de::GPU_Vector<float>& dst);
    }
}


__global__
/**
* @param len : It is in type float4
*/
void cu_general_sigmoid_f(float4* src, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmp;
    float res;

    if (tid < len) {
        tmp = src[tid];

        res = __expf(__fmul_rn(tmp.x, -1.f));
        tmp.x = __frcp_rn(__fadd_rn(1.f, res));

        res = __expf(__fmul_rn(tmp.y, -1.f));
        tmp.y = __frcp_rn(__fadd_rn(1.f, res));

        res = __expf(__fmul_rn(tmp.z, -1.f));
        tmp.z = __frcp_rn(__fadd_rn(1.f, res));

        res = __expf(__fmul_rn(tmp.w, -1.f));
        tmp.w = __frcp_rn(__fadd_rn(1.f, res));

        dst[tid] = tmp;
    }
}


namespace decx
{
    void _dev_sigmoid_f_caller_Mat(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, de::DH* handle);

    void _dev_sigmoid_f_caller_Vec(_GPU_Vector<float>* src, _GPU_Vector<float>* dst, de::DH* handle);
}


void decx::_dev_sigmoid_f_caller_Mat(_GPU_Matrix<float>* src, _GPU_Matrix<float>* dst, de::DH *handle)
{
    if (src->width != dst->width || src->height != dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return;
    }

    const size_t len = decx::utils::ceil<size_t>(
        static_cast<size_t>(src->pitch) * static_cast<size_t>(src->height), 4);


    cu_general_sigmoid_f<<<decx::utils::ceil<size_t>(len, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock>>>(
        reinterpret_cast<float4*>(src->Mat.ptr), reinterpret_cast<float4*>(dst->Mat.ptr), len);
}



void decx::_dev_sigmoid_f_caller_Vec(_GPU_Vector<float>* src, _GPU_Vector<float>* dst, de::DH *handle)
{
    if (src->length != dst->length) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return;
    }

    const size_t len = decx::utils::ceil<size_t>(src->length, 4);

    cu_general_sigmoid_f << < decx::utils::ceil<size_t>(len, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
        reinterpret_cast<float4*>(src->Vec.ptr), reinterpret_cast<float4*>(dst->Vec.ptr), len);
}




de::DH de::nn::sigmoid_f(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst)
{
    de::DH handle;
    decx::Success(&handle);

    _GPU_Matrix<float>* _src = dynamic_cast<_GPU_Matrix<float>*>(&src);
    _GPU_Matrix<float>* _dst = dynamic_cast<_GPU_Matrix<float>*>(&dst);

    decx::_dev_sigmoid_f_caller_Mat(_src, _dst, &handle);

    return handle;
}


de::DH de::nn::sigmoid_f(de::GPU_Vector<float>& src, de::GPU_Vector<float>& dst)
{
    de::DH handle;
    decx::Success(&handle);

    _GPU_Vector<float>* _src = dynamic_cast<_GPU_Vector<float>*>(&src);
    _GPU_Vector<float>* _dst = dynamic_cast<_GPU_Vector<float>*>(&dst);

    decx::_dev_sigmoid_f_caller_Vec(_src, _dst, &handle);

    return handle;
}


#endif
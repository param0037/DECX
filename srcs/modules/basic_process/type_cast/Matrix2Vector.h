/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _MATRIX2vECTOR_H_
#define _MATRIX2vECTOR_H_


#include "../../classes/Matrix.h"
#include "../../classes/GPU_Matrix.h"
#include "../../classes/Vector.h"
#include "../../classes/GPU_Vector.h"


namespace de
{
    namespace cuda
    {
        template <typename T>
        de::DH Matrix_2_Vector(de::GPU_Matrix<T>& src, de::GPU_Vector<T>& dst);


        template <typename T>
        de::DH Matrix_2_Vector(de::GPU_Matrix<T>& src, de::GPU_Vector<T>& dst, const size_t offset);


        template <typename T>
        de::DH Vector_2_Matrix(de::GPU_Vector<T>& src, de::GPU_Matrix<T>& dst);


        template <typename T>
        de::DH Vector_2_Matrix(de::GPU_Vector<T>& src, de::GPU_Matrix<T>& dst, const size_t offset);
    }
}


using decx::_Matrix;
using decx::_GPU_Matrix;
using decx::_Vector;
using decx::_GPU_Vector;


template <typename T>
de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<T>& src, de::GPU_Vector<T>& dst)
{
    _GPU_Matrix<T>* _src = dynamic_cast<_GPU_Matrix<T>*>(&src);
    _GPU_Vector<T>* _dst = dynamic_cast<_GPU_Vector<T>*>(&dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    checkCudaErrors(cudaMemcpy2D(_dst->Vec.ptr, _src->width * sizeof(T),
        _src->Mat.ptr, _src->pitch * sizeof(T), _src->width * sizeof(T), _src->height, cudaMemcpyDeviceToDevice));

    return handle;
}

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<int>& src, de::GPU_Vector<int>& dst);

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<float>& src, de::GPU_Vector<float>& dst);

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<double>& src, de::GPU_Vector<double>& dst);

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<de::Half>& src, de::GPU_Vector<de::Half>& dst);




template <typename T>
de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<T>& src, de::GPU_Vector<T>& dst, const size_t offset)
{
    _GPU_Matrix<T>* _src = dynamic_cast<_GPU_Matrix<T>*>(&src);
    _GPU_Vector<T>* _dst = dynamic_cast<_GPU_Vector<T>*>(&dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    checkCudaErrors(cudaMemcpy2D(_dst->Vec.ptr + offset, _src->width * sizeof(T),
        _src->Mat.ptr, _src->pitch * sizeof(T), _src->width * sizeof(T), _src->height, cudaMemcpyDeviceToDevice));

    return handle;
}

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<int>& src, de::GPU_Vector<int>& dst, const size_t offset);

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<float>& src, de::GPU_Vector<float>& dst, const size_t offset);

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<double>& src, de::GPU_Vector<double>& dst, const size_t offset);

template _DECX_API_ de::DH de::cuda::Matrix_2_Vector(de::GPU_Matrix<de::Half>& src, de::GPU_Vector<de::Half>& dst, const size_t offset);



template <typename T>
de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<T>& src, de::GPU_Matrix<T>& dst)
{
    _GPU_Vector<T>* _src = dynamic_cast<_GPU_Vector<T>*>(&src);
    _GPU_Matrix<T>* _dst = dynamic_cast<_GPU_Matrix<T>*>(&dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    checkCudaErrors(cudaMemcpy2D(_dst->Mat.ptr, _dst->pitch * sizeof(T),
        _src->Vec.ptr, _dst->width * sizeof(T), _dst->width * sizeof(T), _dst->height, cudaMemcpyDeviceToDevice));

    return handle;
}

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<int>& src, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<float>& src, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<double>& src, de::GPU_Matrix<double>& dst);

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<de::Half>& src, de::GPU_Matrix<de::Half>& dst);




template <typename T>
de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<T>& src, de::GPU_Matrix<T>& dst, const size_t offset)
{
    _GPU_Vector<T>* _src = dynamic_cast<_GPU_Vector<T>*>(&src);
    _GPU_Matrix<T>* _dst = dynamic_cast<_GPU_Matrix<T>*>(&dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    checkCudaErrors(cudaMemcpy2D(_dst->Mat.ptr, _dst->pitch * sizeof(T),
        _src->Vec.ptr + offset, _dst->width * sizeof(T), _dst->width * sizeof(T), _dst->height, cudaMemcpyDeviceToDevice));

    return handle;
}

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<int>& src, de::GPU_Matrix<int>& dst, const size_t offset);

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<float>& src, de::GPU_Matrix<float>& dst, const size_t offset);

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<double>& src, de::GPU_Matrix<double>& dst, const size_t offset);

template _DECX_API_ de::DH de::cuda::Vector_2_Matrix(de::GPU_Vector<de::Half>& src, de::GPU_Matrix<de::Half>& dst, const size_t offset);


#endif
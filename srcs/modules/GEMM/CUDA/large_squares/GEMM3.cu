/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CUDA_GEMM3_H_
#define _CUDA_GEMM3_H_

#include "../../../classes/MatrixArray.h"
#include "fp16/GEMM3_fp16_caller.h"
#include "fp32/GEMM3_fp32_caller.h"
#include "../../../core/allocators.h"


using de::MatrixArray;
using decx::_MatrixArray;



namespace decx
{
    namespace cuda
    {
        void GEMM3_fp32(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& dst, de::DH* handle);


        void GEMM3_fp16(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& dst, const int flag, de::DH* handle);
        

        void GEMM3_fp32_ABC(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& C, de::MatrixArray& dst, de::DH* handle);


        void GEMM3_fp16_ABC(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& C, de::MatrixArray& dst, const int flag, de::DH* handle);
    }
}




void decx::cuda::GEMM3_fp32(MatrixArray& A, MatrixArray& B, MatrixArray& dst, de::DH *handle)
{
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(handle);
    }

    decx::_MatrixArray* _A = dynamic_cast<decx::_MatrixArray*>(&A);
    decx::_MatrixArray* _B = dynamic_cast<decx::_MatrixArray*>(&B);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    decx::cuda_stream* S[3] = { NULL, NULL, NULL };
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
        if (S[i] == NULL) {
            Print_Error_Message(4, "CUDA stream access failure\n");
            exit(-1);
        }
    }
    
    decx::sGEMM3_caller(_A, _B, _dst, S, handle);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}




void decx::cuda::GEMM3_fp16(MatrixArray& A, MatrixArray& B, MatrixArray& dst, const int flag, de::DH* handle)
{
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(handle);
    }

    decx::_MatrixArray* _A = dynamic_cast<decx::_MatrixArray*>(&A);
    decx::_MatrixArray* _B = dynamic_cast<decx::_MatrixArray*>(&B);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    decx::cuda_stream* S[3] = { NULL, NULL, NULL };
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
        if (S[i] == NULL) {
            Print_Error_Message(4, "CUDA stream access failure\n");
            exit(-1);
        }
    }

    decx::hGEMM3_caller(_A, _B, _dst, S, flag, handle);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}





void decx::cuda::GEMM3_fp32_ABC(MatrixArray& A, MatrixArray& B, MatrixArray& C, MatrixArray& dst, de::DH* handle)
{
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(handle);
    }

    decx::_MatrixArray* _A = dynamic_cast<decx::_MatrixArray*>(&A);
    decx::_MatrixArray* _B = dynamic_cast<decx::_MatrixArray*>(&B);
    decx::_MatrixArray* _C = dynamic_cast<decx::_MatrixArray*>(&C);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    decx::cuda_stream* S[3] = { NULL, NULL, NULL };
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
        if (S[i] == NULL) {
            Print_Error_Message(4, "CUDA stream access failure\n");
            exit(-1);
        }
    }

    decx::sGEMM3_ABC_caller(_A, _B, _C, _dst, S, handle);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}





void decx::cuda::GEMM3_fp16_ABC(MatrixArray& A, MatrixArray& B, MatrixArray& C, MatrixArray& dst, const int flag, de::DH* handle)
{
    decx::_MatrixArray* _A = dynamic_cast<decx::_MatrixArray*>(&A);
    decx::_MatrixArray* _B = dynamic_cast<decx::_MatrixArray*>(&B);
    decx::_MatrixArray* _C = dynamic_cast<decx::_MatrixArray*>(&C);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    decx::cuda_stream* S[3] = { NULL, NULL, NULL };
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
        if (S[i] == NULL) {
            Print_Error_Message(4, "CUDA stream access failure\n");
            exit(-1);
        }
    }

    decx::hGEMM3_ABC_caller(_A, _B, _C, _dst, S, flag, handle);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH GEMM3(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& dst, const int flag);


        _DECX_API_ de::DH GEMM3(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& C, de::MatrixArray& dst, const int flag);
    }
}



_DECX_API_ 
de::DH de::cuda::GEMM3(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& dst, const int flag)
{
    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(&handle);
    }
    
    switch (A.Type())
    {
    case::decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM3_fp32(A, B, dst, &handle);
        break;

    case::decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM3_fp16(A, B, dst, flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_
de::DH de::cuda::GEMM3(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& C, de::MatrixArray& dst, const int flag)
{
    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::err::CUDA_Not_init(&handle);
    }

    switch (A.Type())
    {
    case::decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM3_fp32_ABC(A, B, C, dst, &handle);
        break;

    case::decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM3_fp16_ABC(A, B, C, dst, flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}


#endif
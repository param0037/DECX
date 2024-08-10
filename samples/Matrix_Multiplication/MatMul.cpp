/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../../includes/DECX.h"
#include <iostream>
#include <iomanip>


#define W 1013
#define H 1077
#define L 1005


void MatMul_CPU()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_FP32_, L, H);
    de::Matrix& B = de::CreateMatrixRef(de::_FP32_, W, L);
    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, W, H);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<float>(i, j) = j;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<float>(i, j) = i;
        }
    }

    // Change the allowed concurrency core invovled in computation
    // Verified by checking the CPU occupancy
    // Chnage the maximum allowed concurrency to 9 (default = the hardware concurrency of your own CPU)
    de::cpu::DecxSetThreadingNum(9);
    for (int i = 0; i < 500; ++i) {
        de::blas::cpu::GEMM(A, B, dst);
    }
    // Chnage the maximum allowed concurrency to 12
    de::cpu::DecxSetThreadingNum(12);
    for (int i = 0; i < 500; ++i) {
        de::blas::cpu::GEMM(A, B, dst);
    }

    for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
            std::cout << std::setw(15) << *dst.ptr<float>(i, j);
        }
        std::cout << std::endl;
    }
}


void MatMul_GPU_CPL32()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_COMPLEX_F32_, L, H);
    de::Matrix& B = de::CreateMatrixRef(de::_COMPLEX_F32_, W, L);
    de::Matrix& C = de::CreateMatrixRef(de::_COMPLEX_F32_, W, H);
    de::Matrix& dst = de::CreateMatrixRef(de::_COMPLEX_F32_, W, H);

    de::GPU_Matrix& dev_A = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, L, H);
    de::GPU_Matrix& dev_B = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, W, L);
    de::GPU_Matrix& dev_C = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, W, H);
    de::GPU_Matrix& dev_dst = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, W, H);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            A.ptr<de::CPf>(i, j)->real = j;
            A.ptr<de::CPf>(i, j)->image = 0;
        }
    }

    for (int i = 0; i < C.Height(); ++i) {
        for (int j = 0; j < C.Width(); ++j) {
            C.ptr<de::CPf>(i, j)->real = 1e6;
            C.ptr<de::CPf>(i, j)->image = 1e6;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            B.ptr<de::CPf>(i, j)->real = i;
            B.ptr<de::CPf>(i, j)->image = 0;
        }
    }

    de::Memcpy(A, dev_A, { 0, 0 }, { 0, 0 }, { A.Width(), A.Height() }, de::DECX_MEMCPY_H2D);
    de::Memcpy(B, dev_B, { 0, 0 }, { 0, 0 }, { B.Width(), B.Height() }, de::DECX_MEMCPY_H2D);
    de::Memcpy(C, dev_C, { 0, 0 }, { 0, 0 }, { C.Width(), C.Height() }, de::DECX_MEMCPY_H2D);

    clock_t s, e;
    s = clock();
    for (int i = 0; i < 1; ++i) {
        de::blas::cuda::GEMM(dev_A, dev_B, dev_C, dev_dst, 0);
    }
    e = clock();

    std::cout << "time : " << (e - s) / 1 << "msec" << std::endl;

    de::Memcpy(dst, dev_dst, { 0, 0 }, { 0, 0 }, { dst.Width(), dst.Height() }, de::DECX_MEMCPY_D2H);

    for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
            std::cout << std::setw(15) << dst.ptr<de::CPf>(i, j)->real;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << std::setw(15) << dst.ptr<de::CPf>(i, j)->image;
        }
        std::cout << std::endl;
    }

    // calculate the supposed result:
    float _sup_real = 0;
    for (int i = 0; i < L; ++i) {
        _sup_real += i * i;
    }
    _sup_real += 1e6;
    float _sup_image = 1e6;

    std::cout << "the supposed result on each element is :" << _sup_real << " + j" << _sup_image << std::endl;
}


int main()
{
    MatMul_GPU_CPL32();
    //MatMul_CPU();

    system("pause");
    return 0;
}
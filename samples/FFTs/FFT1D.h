/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#pragma once

#include <DECX.h>
#include <iostream>
#include <iomanip>


using namespace std;


#define _BENCHMARK_MODE_ 0


static void FFT1D_CPU(const de::_DATA_TYPES_FLAGS_ _inout_type, const uint64_t len)
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Vector& src = de::CreateVectorRef(_inout_type, len);
    de::Vector& dst = de::CreateVectorRef(de::_COMPLEX_F32_, len);
    de::Vector& IFFT_res = de::CreateVectorRef(_inout_type, len);

    for (int i = 0; i < src.Len(); ++i) {
        if (_inout_type == de::_FP32_) {
            *src.ptr_fp32(i) = i;
        }
        else {
            src.ptr_cpl32(i)->real = i;
            src.ptr_cpl32(i)->image = 0;
        }
    }

    clock_t s, e;

    de::cpu::DecxSetThreadingNum(12);

    s = clock();
#if _BENCHMARK_MODE_
    for (int i = 0; i < 1000; ++i) {
#else
    for (int i = 0; i < 1; ++i) {
#endif
        de::dsp::cpu::FFT(src, dst);
        de::dsp::cpu::IFFT(dst, IFFT_res, de::_FP32_);
    }
    e = clock();

    for (int i = 0; i < 30; ++i) {
        if (_inout_type == de::_FP32_) {
            std::cout << i << ": " << (int)*src.ptr_fp32(i) << endl;
        }
        else {
            std::cout << i << ": " << src.ptr_cpl32(i)->real << ", " << src.ptr_cpl32(i)->image << endl;
        }
    }
    std::cout << "\n";
    for (int i = len - 30; i < len; ++i) {
        std::cout << i << ": " << dst.ptr_cpl32(i)->real << ", " << dst.ptr_cpl32(i)->image << endl;
    }
    std::cout << "\n";
    for (int i = len - 300; i < len; ++i) {
        if (_inout_type == de::_FP32_) {
            std::cout << i << ": " << (int)*IFFT_res.ptr_fp32(i) << endl;
        }
        else {
            std::cout << i << ": " << IFFT_res.ptr_cpl32(i)->real << ", " << IFFT_res.ptr_cpl32(i)->image << endl;
        }
    }
    std::cout << "\n";
#if _BENCHMARK_MODE_
    std::cout << "time spent : " << (double)(e - s) / (1000 * 2.0) << " msec" << endl;
#else
    std::cout << "time spent : " << (double)(e - s) / (2.0) << " msec" << endl;
#endif

    src.release();
    dst.release();
    IFFT_res.release();
}




static void FFT1D_CUDA(const de::_DATA_TYPES_FLAGS_ _inout_type, const uint64_t len)
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Vector& src = de::CreateVectorRef(_inout_type, len);
    de::Vector& dst = de::CreateVectorRef(de::_COMPLEX_F32_, len);
    de::Vector& IFFT_res = de::CreateVectorRef(_inout_type, len);

    de::GPU_Vector& Dsrc = de::CreateGPUVectorRef(_inout_type, len);
    de::GPU_Vector& Ddst = de::CreateGPUVectorRef(de::_COMPLEX_F32_, len);
    de::GPU_Vector& DIFFT_res = de::CreateGPUVectorRef(_inout_type, len);

    for (int i = 0; i < src.Len(); ++i) {
        if (_inout_type == de::_FP32_) {
            *src.ptr_fp32(i) = i;
        }
        else {
            src.ptr_cpl32(i)->real = i;
            src.ptr_cpl32(i)->image = 0;
        }
    }

    de::Memcpy(src, Dsrc, 0, 0, src.Len(), de::DECX_MEMCPY_H2D);

    clock_t s, e;

    s = clock();
#if _BENCHMARK_MODE_
    for (int i = 0; i < 1000; ++i) {
#else
    for (int i = 0; i < 1; ++i) {
#endif
        /**
        * It is highly recommended to pass a GPU_* to a cuda operator, if possible,
        * avoiding data transmissions between host and device.
        */
        de::dsp::cuda::FFT(Dsrc, Ddst);
        de::dsp::cuda::IFFT(Ddst, DIFFT_res, de::_FP32_);
    }
    e = clock();

    de::Memcpy(dst, Ddst, 0, 0, dst.Len(), de::DECX_MEMCPY_D2H);
    de::Memcpy(IFFT_res, DIFFT_res, 0, 0, IFFT_res.Len(), de::DECX_MEMCPY_D2H);

    for (int i = 0; i < 30; ++i) {
        if (_inout_type == de::_FP32_) {
            std::cout << i << ": " << (int)*src.ptr_fp32(i) << endl;
        }
        else {
            std::cout << i << ": " << src.ptr_cpl32(i)->real << ", " << src.ptr_cpl32(i)->image << endl;
        }
    }
    std::cout << "\n";
    for (int i = len - 30; i < len; ++i) {
        std::cout << i << ": " << dst.ptr_cpl32(i)->real << ", " << dst.ptr_cpl32(i)->image << endl;
    }
    std::cout << "\n";
    for (int i = len - 300; i < len; ++i) {
        if (_inout_type == de::_FP32_) {
            std::cout << i << ": " << (int)*IFFT_res.ptr_fp32(i) << endl;
        }
        else {
            std::cout << i << ": " << IFFT_res.ptr_cpl32(i)->real << ", " << IFFT_res.ptr_cpl32(i)->image << endl;
        }
    }
    std::cout << "\n";
#if _BENCHMARK_MODE_
    std::cout << "time spent : " << (double)(e - s) / (1000 * 2.0) << " msec" << endl;
#else
    std::cout << "time spent : " << (double)(e - s) / (2.0) << " msec" << endl;
#endif

    src.release();
    dst.release();
    IFFT_res.release();

    Dsrc.release();
    Ddst.release();
    DIFFT_res.release();
    }

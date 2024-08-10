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
            *src.ptr<float>(i) = i;
        }
        else {
            src.ptr<de::CPf>(i)->real = i;
            src.ptr<de::CPf>(i)->image = 0;
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
            std::cout << i << ": " << (int)*src.ptr<float>(i) << endl;
        }
        else {
            std::cout << i << ": " << src.ptr<de::CPf>(i)->real << ", " << src.ptr<de::CPf>(i)->image << endl;
        }
    }
    std::cout << "\n";
    for (int i = len - 30; i < len; ++i) {
        std::cout << i << ": " << dst.ptr<de::CPf>(i)->real << ", " << dst.ptr<de::CPf>(i)->image << endl;
    }
    std::cout << "\n";
    for (int i = len - 300; i < len; ++i) {
        if (_inout_type == de::_FP32_) {
            std::cout << i << ": " << (int)*IFFT_res.ptr<float>(i) << endl;
        }
        else {
            std::cout << i << ": " << IFFT_res.ptr<de::CPf>(i)->real << ", " << IFFT_res.ptr<de::CPf>(i)->image << endl;
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
            *src.ptr<float>(i) = i;
        }
        else {
            src.ptr<de::CPf>(i)->real = i;
            src.ptr<de::CPf>(i)->image = 0;
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
            std::cout << i << ": " << (int)*src.ptr<float>(i) << endl;
        }
        else {
            std::cout << i << ": " << src.ptr<de::CPf>(i)->real << ", " << src.ptr<de::CPf>(i)->image << endl;
        }
    }
    std::cout << "\n";
    for (int i = len - 30; i < len; ++i) {
        std::cout << i << ": " << dst.ptr<de::CPf>(i)->real << ", " << dst.ptr<de::CPf>(i)->image << endl;
    }
    std::cout << "\n";
    for (int i = len - 300; i < len; ++i) {
        if (_inout_type == de::_FP32_) {
            std::cout << i << ": " << (int)*IFFT_res.ptr<float>(i) << endl;
        }
        else {
            std::cout << i << ": " << IFFT_res.ptr<de::CPf>(i)->real << ", " << IFFT_res.ptr<de::CPf>(i)->image << endl;
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

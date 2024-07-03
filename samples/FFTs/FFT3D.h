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



static void FFT3D_CPU(const de::_DATA_TYPES_FLAGS_ _inout_type, const de::Point3D dims_DWH)
{
    de::InitCuda();
    de::InitCPUInfo();
    //W   H   D

    de::Tensor& src = de::CreateTensorRef(_inout_type, dims_DWH.y, dims_DWH.z, dims_DWH.x);
    de::Tensor& dst = de::CreateTensorRef(de::_COMPLEX_F32_, dims_DWH.y, dims_DWH.z, dims_DWH.x);

    for (int32_t i = 0; i < src.Height(); ++i) {
        for (int32_t j = 0; j < src.Width(); ++j) {
            for (int32_t k = 0; k < src.Depth(); ++k)
            {
                if (_inout_type == de::_FP32_) {
                    *src.ptr<float>(i, j, k) = -j + k + i;
                }
                else {
                    src.ptr<de::CPf>(i, j, k)->real = -j + k + i;
                    src.ptr<de::CPf>(i, j, k)->image = 0;
                }
            }
        }
    }
    de::dsp::cpu::FFT(src, dst, de::_COMPLEX_F32_);

    int _show_dex = 9;

    cout << "\noriginal\n";
    for (uint32_t j = 0; j < src.Width(); ++j)
    {
        if (_inout_type == de::_FP32_) {
            cout << *src.ptr<float>(_show_dex, j, _show_dex) << '\n';
        }
        else {
            cout << src.ptr<de::CPf>(_show_dex, j, _show_dex)->real << ", " << src.ptr<de::CPf>(_show_dex, j, _show_dex)->image << '\n';
        }
    }
    cout << "\nspectrum\n";

    for (uint32_t k = 0; k < src.Width(); ++k) {
        cout << dst.ptr<de::CPf>(_show_dex, k, _show_dex)->real << ", " << dst.ptr<de::CPf>(_show_dex, k, _show_dex)->image << '\n';
    }

    de::dsp::cpu::IFFT(dst, src, _inout_type);

    cout << "\nrecovered\n";;

    for (uint32_t j = 0; j < src.Width(); ++j)
    {
        if (_inout_type == de::_FP32_) {
            cout << *src.ptr<float>(_show_dex, j, _show_dex) << '\n';
        }
        else {
            cout << src.ptr<de::CPf>(_show_dex, j, _show_dex)->real << ", " << src.ptr<de::CPf>(_show_dex, j, _show_dex)->image << '\n';
        }
    }
}




static void FFT3D_CUDA(const de::_DATA_TYPES_FLAGS_ _inout_type, const de::Point3D dims_DWH)
{
    de::InitCuda();
    de::InitCPUInfo();
    //W   H   D

    de::Tensor& src = de::CreateTensorRef(_inout_type, dims_DWH.y, dims_DWH.z, dims_DWH.x);
    de::Tensor& dst = de::CreateTensorRef(de::_COMPLEX_F32_, dims_DWH.y, dims_DWH.z, dims_DWH.x);

    de::GPU_Tensor& Dsrc = de::CreateGPUTensorRef(_inout_type, src.Width(), src.Height(), src.Depth());
    de::GPU_Tensor& Ddst = de::CreateGPUTensorRef(de::_COMPLEX_F32_, src.Width(), src.Height(), src.Depth());
    de::GPU_Tensor& Drecover = de::CreateGPUTensorRef(_inout_type, src.Width(), src.Height(), src.Depth());

    for (int32_t i = 0; i < src.Height(); ++i) {
        for (int32_t j = 0; j < src.Width(); ++j) {
            for (int32_t k = 0; k < src.Depth(); ++k)
            {
                if (_inout_type == de::_FP32_) {
                    *src.ptr<float>(i, j, k) = -j + k + i;
                }
                else {
                    src.ptr<de::CPf>(i, j, k)->real = -j + k + i;
                    src.ptr<de::CPf>(i, j, k)->image = 0;
                }
            }
        }
    }

    de::Memcpy(src, Dsrc, de::Point3D(0, 0, 0), de::Point3D(0, 0, 0), de::Point3D(src.Depth(), src.Width(), src.Height()), de::DECX_MEMCPY_H2D);

    de::dsp::cuda::FFT(Dsrc, Ddst, de::_COMPLEX_F32_);

    int _show_dex = 9;

    cout << "\noriginal\n";
    for (uint32_t j = 0; j < src.Width(); ++j)
    {
        if (_inout_type == de::_FP32_) {
            cout << *src.ptr<float>(_show_dex, j, _show_dex) << '\n';
        }
        else {
            cout << src.ptr<de::CPf>(_show_dex, j, _show_dex)->real << ", " << src.ptr<de::CPf>(_show_dex, j, _show_dex)->image << '\n';
        }
    }
    cout << "\nspectrum\n";

    de::Memcpy(dst, Ddst, de::Point3D(0, 0, 0), de::Point3D(0, 0, 0), de::Point3D(src.Depth(), src.Width(), src.Height()), de::DECX_MEMCPY_D2H);

    for (uint32_t k = 0; k < src.Width(); ++k) {
        cout << dst.ptr<de::CPf>(_show_dex, k, _show_dex)->real << ", " << dst.ptr<de::CPf>(_show_dex, k, _show_dex)->image << '\n';
    }

    de::dsp::cuda::IFFT(Ddst, Drecover, _inout_type);

    de::Memcpy(src, Drecover, de::Point3D(0, 0, 0), de::Point3D(0, 0, 0), de::Point3D(src.Depth(), src.Width(), src.Height()), de::DECX_MEMCPY_D2H);


    cout << "\nrecovered\n";;

    for (uint32_t j = 0; j < src.Width(); ++j)
    {
        if (_inout_type == de::_FP32_) {
            cout << *src.ptr<float>(_show_dex, j, _show_dex) << '\n';
        }
        else {
            cout << src.ptr<de::CPf>(_show_dex, j, _show_dex)->real << ", " << src.ptr<de::CPf>(_show_dex, j, _show_dex)->image << '\n';
        }
    }
}

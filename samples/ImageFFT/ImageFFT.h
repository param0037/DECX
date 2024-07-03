/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _IMAGE_FFT_H_
#define _IMAGE_FFT_H_


#include "../../includes/DECX.h"
#include <iostream>


void decx_frequnce_domain_filtering_CUDA()
{
    de::InitCPUInfo();
    de::InitCuda();

    // create an empty de::Matrix object
    de::Matrix& src = de::CreateMatrixRef();
    // load image data from hard drive to this empty de::matrix object
    de::vis::ReadImage("../../ImageFFT/test_FFT2.jpg", src);
    // create a de::Matrix object to store data of grayscale image
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    // convert a BGR image to grayscale
    de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);

    // create a de::Matrix object to store data for the input of FFT2D
    de::Matrix& src_fp32 = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height());
    // create a de::GPU_Matrix object to store data for the input of FFT2D
    de::GPU_Matrix& D_src_fp32 = de::CreateGPUMatrixRef(de::_FP32_, src.Width(), src.Height());

    de::GPU_Matrix& D_FFTres = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
    de::GPU_Matrix& D_Filter_res = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
    de::Matrix& FFTres = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());

    de::Matrix& FFT_res_mod = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height());
    de::Matrix& FFT_illustration = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::Matrix& dst_img_recover = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());

    // convert data_type from uint8(unsigned char) to fp32(32-bit floating point)
    de::cpu::TypeCast(src_gray, src_fp32, de::CVT_UINT8_FP32);
    // load data from host to device
    de::Memcpy(src_fp32, D_src_fp32, { 0, 0 }, { 0, 0 }, { src_fp32.Width(), src_fp32.Height() }, de::DECX_MEMCPY_H2D);
    // call FFT2D() API on GPU
    de::dsp::cuda::FFT(D_src_fp32, D_FFTres, de::_COMPLEX_F32_);

    //de::signal::cuda::Gaussian_Window2D(D_FFTres, D_Filter_res, de::Point2D_f(0, 0), de::Point2D_f(50, 200), 0.9);
    // call low-pass filter API on GPU
    de::dsp::cuda::LowPass2D_Ideal(D_FFTres, D_Filter_res, de::Point2D(100, 50));
    // load data from device to host
    de::Memcpy(FFTres, D_Filter_res, { 0, 0 }, { 0, 0 }, { FFTres.Width(), FFTres.Height() }, de::DECX_MEMCPY_D2H);

    clock_t s, e;
    s = clock();
    // call IFFT2D() API on GPU
    de::dsp::cuda::IFFT(D_Filter_res, D_src_fp32, de::_FP32_);
    e = clock();
    // load data from device to host
    de::Memcpy(src_fp32, D_src_fp32, { 0, 0 }, { 0, 0 }, { src_fp32.Width(), src_fp32.Height() }, de::DECX_MEMCPY_D2H);

    std::cout << "time spent (IFFT2D) on GPU : " << (e - s) << "msec" << std::endl;
    // get the amplitudes specturm of the filtered image
    de::dsp::cpu::Module(FFTres, FFT_res_mod);

    // visualize the amplitudes specturm of the filtered image
    for (int i = 0; i < src.Height(); ++i) {
        for (int j = 0; j < src.Width(); ++j) {
            *FFT_illustration.ptr<uint8_t>(i, j) = *FFT_res_mod.ptr<float>(i, j) / 6e5;
        }
    }

    // convert data type from fp32(32-bit floating point) to uint8(unsigned char), (saturated cast)
    de::cpu::TypeCast(src_fp32, dst_img_recover, de::CVT_FP32_UINT8 | de::CVT_UINT8_SATURATED);

    de::vis::ShowImg(src_gray, "original image");
    de::vis::ShowImg(FFT_illustration, "filter result (on frequency domain)");
    de::vis::ShowImg(dst_img_recover, "filtered image");
    de::vis::wait_untill_quit();

    // release the objects
    src.release();
    src_gray.release();
    src_fp32.release();
    FFT_res_mod.release();
    FFT_illustration.release();
    dst_img_recover.release();

    D_src_fp32.release();
    D_Filter_res.release();
    D_FFTres.release();

    system("pause");
}


void decx_frequnce_domain_filtering_CPU()
{
    de::InitCPUInfo();
    de::InitCuda();

    // create an empty de::Matrix object
    de::Matrix& src = de::CreateMatrixRef();
    // load image data from hard drive to this empty de::matrix object
    de::vis::ReadImage("../../ImageFFT/test_FFT2.jpg", src);
    //de::vis::ReadImage("E:/User/User.default/Pictures/1280_al.jpg", src);
    // create a de::Matrix object to store data of grayscale image
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    // convert a BGR image to grayscale
    de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);

    de::Matrix& FFTres = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
    de::Matrix& Filter_res = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());

    de::Matrix& FFT_res_mod = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height());
    de::Matrix& FFT_illustration = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::Matrix& dst_img_recover = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());

    // call FFT2D() API on GPU
    de::dsp::cpu::FFT(src_gray, FFTres, de::_COMPLEX_F32_);

    // call low-pass filter API on GPU
    //de::dsp::cpu::LowPass2D_Ideal(FFTres, Filter_res, de::Point2D(100, 50));

    clock_t s, e;
    s = clock();
    // call IFFT2D() API on GPU
    de::dsp::cpu::IFFT(FFTres, dst_img_recover, de::_UINT8_);
    e = clock();

    std::cout << "time spent (IFFT2D) on CPU : " << (e - s) << "msec" << std::endl;
    // get the amplitudes specturm of the filtered image
    de::dsp::cpu::Module(FFTres, FFT_res_mod);

    // visualize the amplitudes specturm of the filtered image
    for (int i = 0; i < src.Height(); ++i) {
        for (int j = 0; j < src.Width(); ++j) {
            *FFT_illustration.ptr<uint8_t>(i, j) = *FFT_res_mod.ptr<float>(i, j) / 1e2;
        }
    }
    float _scalar = 1e2;
    de::cpu::Div(FFT_res_mod, &_scalar, FFT_res_mod);
    de::cpu::TypeCast(FFT_res_mod, FFT_illustration, de::CVT_FP32_UINT8 | de::CVT_UINT8_CLAMP_TO_ZERO | de::CVT_UINT8_SATURATED);

    de::vis::ShowImg(src_gray, "original image");
    de::vis::ShowImg(FFT_illustration, "filter result (on frequency domain)");
    de::vis::ShowImg(dst_img_recover, "filtered image");
    de::vis::wait_untill_quit();

    // release the objects
    src.release();
    src_gray.release();
    FFT_res_mod.release();
    FFT_illustration.release();
    dst_img_recover.release();

    system("pause");
}



#endif
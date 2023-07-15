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


#include <DECX.h>
#include <iostream>

void decx_frequnce_domain_filtering_CUDA()
{
    de::InitCPUInfo();
    de::InitCuda();

    // create an empty de::Matrix object
    de::Matrix& src = de::CreateMatrixRef();
    // load image data from hard drive to this empty de::matrix object, <store_type = de::page_default>
    de::vis::ReadImage("../../ImageFFT/test_FFT2.jpg", src);
    // create a de::Matrix object to store data of grayscale image
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), de::Page_Default);
    // convert a BGR image to grayscale
    de::vis::merge_channel(src, src_gray, de::vis::BGR_to_Gray);

    // create a de::Matrix object to store data for the input of FFT2D
    de::Matrix& src_fp32 = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height(), de::Page_Locked);
    // create a de::GPU_Matrix object to store data for the input of FFT2D
    de::GPU_Matrix& D_src_fp32 = de::CreateGPUMatrixRef(de::_FP32_, src.Width(), src.Height());

    de::GPU_Matrix& D_FFTres = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
    de::GPU_Matrix& D_Filter_res = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
    de::Matrix& FFTres = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height(), de::Page_Locked);

    de::Matrix& FFT_res_mod = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height(), de::Page_Default);
    de::Matrix& FFT_illustration = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), de::Page_Default);
    de::Matrix& dst_img_recover = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), de::Page_Default);

    // convert data_type from uint8(unsigned char) to fp32(32-bit floating point)
    de::cpu::TypeCast(src_gray, src_fp32, de::CVT_UINT8_FP32);
    // load data from host to device
    de::MemcpyLinear(src_fp32, D_src_fp32, de::DECX_MEMCPY_H2D);
    // call FFT2D() API on GPU
    de::signal::cuda::FFT2D(D_src_fp32, D_FFTres, de::signal::FFT_R2C);

    //de::signal::cuda::Gaussian_Window2D(D_FFTres, D_Filter_res, de::Point2D_f(0, 0), de::Point2D_f(50, 200), 0.9);
    // call low-pass filter API on GPU
    de::signal::cuda::LowPass2D_Ideal(D_FFTres, D_Filter_res, de::Point2D(100, 50));
    // load data from device to host
    de::MemcpyLinear(FFTres, D_Filter_res, de::DECX_MEMCPY_D2H);

    clock_t s, e;
    s = clock();
    // call IFFT2D() API on GPU
    de::signal::cuda::IFFT2D(D_Filter_res, D_src_fp32, de::signal::IFFT_C2R);
    e = clock();
    // load data from device to host
    de::MemcpyLinear(src_fp32, D_src_fp32, de::DECX_MEMCPY_D2H);

    std::cout << "time spent (IFFT2D) on GPU : " << (e - s) << "msec" << std::endl;
    // get the amplitudes specturm of the filtered image
    de::signal::cpu::Module(FFTres, FFT_res_mod);

    // visualize the amplitudes specturm of the filtered image
    for (int i = 0; i < src.Height(); ++i) {
        for (int j = 0; j < src.Width(); ++j) {
            *FFT_illustration.ptr_uint8(i, j) = *FFT_res_mod.ptr_fp32(i, j) / 6e5;
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
    // load image data from hard drive to this empty de::matrix object, <store_type = de::page_default>
    de::vis::ReadImage("../../ImageFFT/test_FFT2.jpg", src);
    // create a de::Matrix object to store data of grayscale image
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), de::Page_Default);
    // convert a BGR image to grayscale
    de::vis::merge_channel(src, src_gray, de::vis::BGR_to_Gray);

    // create a de::Matrix object to store data for the input of FFT2D
    de::Matrix& src_fp32 = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height(), de::Page_Default);
    // create a de::GPU_Matrix object to store data for the input of FFT2D

    de::Matrix& FFTres = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height(), de::Page_Default);
    de::Matrix& Filter_res = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height(), de::Page_Default);

    de::Matrix& FFT_res_mod = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height(), de::Page_Default);
    de::Matrix& FFT_illustration = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), de::Page_Default);
    de::Matrix& dst_img_recover = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), de::Page_Default);

    // convert data_type from uint8(unsigned char) to fp32(32-bit floating point)
    de::cpu::TypeCast(src_gray, src_fp32, de::CVT_UINT8_FP32);

    // call FFT2D() API on GPU
    de::signal::cpu::FFT2D(src_fp32, FFTres, de::signal::FFT_R2C);

    //de::signal::cuda::Gaussian_Window2D(D_FFTres, D_Filter_res, de::Point2D_f(0, 0), de::Point2D_f(50, 200), 0.9);
    // call low-pass filter API on GPU
    de::signal::cpu::LowPass2D_Ideal(FFTres, Filter_res, de::Point2D(100, 50));

    clock_t s, e;
    s = clock();
    // call IFFT2D() API on GPU
    de::signal::cpu::IFFT2D(Filter_res, src_fp32, de::signal::IFFT_C2R);
    e = clock();

    std::cout << "time spent (IFFT2D) on CPU : " << (e - s) << "msec" << std::endl;
    // get the amplitudes specturm of the filtered image
    de::signal::cpu::Module(FFTres, FFT_res_mod);

    // visualize the amplitudes specturm of the filtered image
    for (int i = 0; i < src.Height(); ++i) {
        for (int j = 0; j < src.Width(); ++j) {
            *FFT_illustration.ptr_uint8(i, j) = *FFT_res_mod.ptr_fp32(i, j) / 6e5;
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

    system("pause");
}



#endif
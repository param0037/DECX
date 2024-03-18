/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include <DECX.h>
#include <iostream>
#include <iomanip>

using namespace std;


struct uchar4 {
    uint8_t x, y, z, w;
};


void Conv2D_CUDA()
{
    de::InitCPUInfo();
    de::InitCuda();

    const uint32_t src_channel = 16;

    de::Matrix& img = de::CreateMatrixRef(/*de::_UCHAR4_, 256, 256*/);
    de::vis::ReadImage("E:/User/User.default/Pictures/lena.jpg", img);
    de::Matrix& gray_img = de::CreateMatrixRef(de::_UINT8_, img.Width(), img.Height());
    de::vis::ColorTransform(img, gray_img, de::vis::RGB_to_Gray);

    de::Tensor& src = de::CreateTensorRef(de::_FP32_, img.Width(), img.Height(), src_channel);
    de::TensorArray& kernel = de::CreateTensorArrayRef(de::_FP32_, 5, 5, src_channel, 4);

    de::GPU_Tensor& D_src = de::CreateGPUTensorRef(de::_FP32_, img.Width(), img.Height(), src_channel);
    de::GPU_TensorArray& D_kernel = de::CreateGPUTensorArrayRef(de::_FP32_, 5, 5, src_channel, 4);

    //de::GPU_Tensor& D_dst = de::CreateGPUTensorRef(de::_FP32_, (img.Width() - 4) / 1, (img.Height() - 4) / 1, 4);
    de::GPU_Tensor& D_dst = de::CreateGPUTensorRef();
    //de::Tensor& dst = de::CreateTensorRef(de::_FP32_, D_dst.Width(), D_dst.Height(), 4);


    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            uchar4 tmp;
            *((float*)&tmp) = *img.ptr_fp32(i, j); //2381440000
            *src.ptr_fp32(i, j, 0) = tmp.x;
            *src.ptr_fp32(i, j, 1) = tmp.y;
            *src.ptr_fp32(i, j, 2) = tmp.z;
        }
    }

    for (int i = 0; i < kernel.Height(); ++i) {
        for (int j = 0; j < kernel.Width(); ++j) {
            for (int k = 0; k < kernel.Depth(); ++k) {
                *kernel.ptr_fp32(i, j, k, 0) = 1.f / (25 * 3);
                *kernel.ptr_fp32(i, j, k, 1) = 1.f / (25 * 3);
                *kernel.ptr_fp32(i, j, k, 2) = 1.f / (25 * 3);
                *kernel.ptr_fp32(i, j, k, 3) = 0;
            }
        }
    }

    de::Memcpy(src, D_src, { 0, 0, 0 }, { 0, 0, 0 }, { src.Depth(), src.Width(), src.Height() }, de::DECX_MEMCPY_H2D);

    de::Tensor& slice = de::CreateTensorRef();
    de::GPU_Tensor& Dslice = de::CreateGPUTensorRef();
    for (int i = 0; i < kernel.TensorNum(); ++i) {
        kernel.Extract_SoftCopy(i, slice);
        D_kernel.Extract_SoftCopy(i, Dslice);
        de::Memcpy(slice, Dslice, { 0, 0, 0 }, { 0, 0, 0 }, { kernel.Depth(), kernel.Width(), kernel.Height() }, de::DECX_MEMCPY_H2D);
    }

    de::nn::cuda::Conv2D(D_src, D_kernel, D_dst, { 1, 1 }, de::_EXTEND_CONSTANT_);

    de::Tensor& dst = de::CreateTensorRef(de::_FP32_, D_dst.Width(), D_dst.Height(), 4);
    de::Matrix& dst_img = de::CreateMatrixRef(de::_UCHAR4_, D_dst.Width(), D_dst.Height());

    de::Memcpy(dst, D_dst, { 0, 0, 0 }, { 0, 0, 0 }, { dst.Depth(), dst.Width(), dst.Height() }, de::DECX_MEMCPY_D2H);

    float* _ptr = dst.ptr_fp32(0, 0, 0);

    for (int i = 0 * 4; i < 10 * 4; ++i) {
        printf("[%d] : %f\n", i, _ptr[i]);
    }
    cout << endl;
    for (int i = (dst.Width() - 40) * 4; i < dst.Width() * 4; ++i) {
        printf("[%d] : %f\n", i, _ptr[i]);
    }

    for (int i = 0; i < dst.Height(); ++i) {
        for (int j = 0; j < dst.Width(); ++j) {
            uchar4 tmp;
            tmp.x = *dst.ptr_fp32(i, j, 0);
            tmp.y = *dst.ptr_fp32(i, j, 1);
            tmp.z = *dst.ptr_fp32(i, j, 2);
            tmp.w = 255;
            *dst_img.ptr_fp32(i, j) = *((float*)&tmp);
        }
    }

    de::vis::ShowImg(img, "1");
    de::vis::ShowImg(dst_img, "2");
    de::vis::wait_untill_quit();

    img.release();
    gray_img.release();
    src.release();
    kernel.release();
    D_src.release();
    D_kernel.release();
    D_dst.release();
}


int main()
{
    Conv2D_CUDA();

    system("pause");
    return 0;
}
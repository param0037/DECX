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


void FindEdge()
{
    de::InitCPUInfo();

    de::Matrix& src = de::CreateMatrixRef();
    de::vis::ReadImage("../../CV_Image_Proc/test_image.jpg", src);
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), 1);
    de::vis::merge_channel(src, src_gray, de::vis::BGR_to_Gray);
    de::Matrix& src_gray_blur = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), 1);
    de::Matrix& dst = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height(), 1);
    //de::Matrix& kernel = de::CreateMatrixRef(de::_FP32_, 5, 5, 1);
    
    //de::cpu::Filter2D(src_gray, kernel, src_gray_blur, de::conv_property::de_conv_zero_compensate, de::_UINT8_);
    de::vis::cpu::Gaussian_Filter(src_gray, src_gray_blur, de::Point2D(11, 11), de::Point2D_f(1, 1), de::_EXTEND_REFLECT_);
    //de::vis::cpu::Bilateral_Filter(src_gray, src_gray_blur, de::Point2D(5, 5), 4, 30, de::_EXTEND_REFLECT_);
    //de::vis::cuda::NLM_Gray(src_gray, src_gray_blur, 9, 2, 10);
    
    //de::vis::cpu::Find_Edge(src_gray_blur, dst, 0, 0, de::vis::DE_SOBEL);
    clock_t s, e;
    s = clock();
    de::vis::cpu::Find_Edge(src_gray, dst, 30, 90, de::vis::DE_SOBEL);
    e = clock();

    std::cout << "time spent : " << e - s << "msec" << std::endl;

    de::vis::ShowImg(src_gray_blur, "1");
    de::vis::ShowImg(dst, "2");
    de::vis::wait_untill_quit();
}


void NLM()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& src = de::CreateMatrixRef();
    de::vis::ReadImage("../../CV_Image_Proc/test_image.jpg", src);
    de::Matrix& dst = de::CreateMatrixRef(de::_UCHAR4_, src.Width(), src.Height(), 0);
    //de::vis::cuda::NLM_RGB_keep_alpha(src, dst, 7, 2, 15);
    clock_t s, e;
    s = clock();
    de::vis::cuda::NLM_RGB_keep_alpha(src, dst, 4, 1, 10);
    e = clock();
    std::cout << "time spent :" << (e - s) << "msec" << std::endl;
    de::vis::ShowImg(src, "src");
    de::vis::ShowImg(dst, "dst");
    de::vis::wait_untill_quit();
}


int main()
{
    //FindEdge();
    NLM();

    system("pause");
    return 0;
}
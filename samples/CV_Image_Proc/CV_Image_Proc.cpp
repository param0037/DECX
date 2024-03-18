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
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);
    de::Matrix& src_gray_blur = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::Matrix& dst = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    
    de::vis::cpu::Gaussian_Filter(src_gray, src_gray_blur, de::Point2D(11, 11), de::Point2D_f(1, 1), de::_EXTEND_REFLECT_);
    
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
    de::Matrix& dst = de::CreateMatrixRef(de::_UCHAR4_, src.Width(), src.Height());
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
    FindEdge();
    //NLM();

    system("pause");
    return 0;
}
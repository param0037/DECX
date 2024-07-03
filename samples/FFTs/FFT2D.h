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


#include "../../includes/DECX.h"
#include <iostream>
#include <iomanip>


static void DECX_FFT2D_CPU()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& src = de::CreateMatrixRef();
    //de::vis::ReadImage("E:/User/User.default/Pictures/1280_al.jpg", src);
    de::vis::ReadImage("E:/User/User.default/Pictures/625_al.jpg", src);
    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::Matrix& dst_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::Matrix& spectrum_visual = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);

    de::Matrix& spectrum = de::CreateMatrixRef(de::_COMPLEX_F64_, src.Width(), src.Height());
    de::GPU_Matrix& Dspectrum = de::CreateGPUMatrixRef(de::_COMPLEX_F64_, src.Width(), src.Height());

    de::Matrix& dst_img_recover = de::CreateMatrixRef(/*de::_UINT8_, src.Width(), src.Height()*/);
    de::GPU_Matrix& Ddst_img_recover = de::CreateGPUMatrixRef(de::_UINT8_, src.Width(), src.Height());

    clock_t s, e;
    s = clock();
    
    de::dsp::cpu::FFT(src_gray, spectrum, de::_COMPLEX_F32_);

    de::dsp::cpu::IFFT(spectrum, dst_img_recover, de::_UINT8_);

    e = clock();
    // Visualize the spectrum
    for (int i = 0; i < src.Height(); ++i) {
        for (int j = 0; j < src.Width(); ++j) {
            *spectrum_visual.ptr<uint8_t>(i, j) =
                abs(spectrum.ptr<de::CPf>(i, j)->real) + abs(spectrum.ptr<de::CPf>(i, j)->image) / 500;
        }
    }

    cout << "time spent : " << (e - s) / (2.0) << " msec" << endl;

    de::vis::ShowImg(src_gray, "original");
    de::vis::ShowImg(spectrum_visual, "spectrum");
    de::vis::ShowImg(dst_gray, "recovered");
    de::vis::wait_untill_quit();
}
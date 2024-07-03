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

#include "FFT1D.h"
#include "FFT3D.h"


int main()
{
    int _dimension = 1;
    cout << "Please select the dimension, 1 for FFT1D; 3 for FFT3D\n";
    cin >> _dimension;

    int mode = 0;
    cout << "Please select the device, 0 for CPU; 1 for GPU\n";
    cin >> mode;
    if (_dimension == 3) {
        if (mode) {
            FFT3D_CUDA(de::_FP32_, { 60, 180, 60 });
        }
        else {
            FFT3D_CPU(de::_FP32_, { 60, 180, 60 });
        }
    }
    else {
        if (mode) {
            FFT1D_CUDA(de::_FP32_, 625 * 625 * 3 * 2);
        }
        else {
            FFT1D_CPU(de::_FP32_, 625 * 625 * 3 * 2);
        }
    }
    system("pause");
    return 0;
}
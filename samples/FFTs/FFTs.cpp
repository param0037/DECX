/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

/*
* For FFT2D please refer to project ImageFFT
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
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT1D.h"
#include "FFT3D.h"


int main()
{
    int _dimension = 1;

    int mode = 0;
    cout << "Please select the device, 0 for CPU; 1 for GPU\n";
    cin >> mode;

    if (mode) {
        //FFT1D_CUDA(de::_FP32_, 625 * 625 * 3 * 2);
    }
    else {
        FFT3D_CPU(de::_FP32_, 625 * 625 * 3 * 2);
    }
    system("pause");
    return 0;
}
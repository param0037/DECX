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


#include "Gaussian_kernel.h"



decx::vis::gaussian_kernel1D::gaussian_kernel1D(const uint32_t ker_length, const float sigma, bool central, const float mean)
{
    if (central) {
        this->_ker_length = ker_length;
        this->_mean = ker_length / 2;
        this->_sigma = sigma;
    }
    else {
        this->_ker_length = ker_length;
        this->_mean = mean;
        this->_sigma = sigma;
    }
    if (decx::alloc::_host_virtual_page_malloc(&this->_kernel_data, this->_ker_length * sizeof(float))) {
        exit(-1);
    }
}



void decx::vis::gaussian_kernel1D::assign(
    decx::PtrInfo<float> kernel_data, const uint32_t ker_length, const float sigma, bool central, const float mean)
{
    if (central) {
        this->_kernel_data = kernel_data;
        this->_ker_length = ker_length;
        this->_mean = ker_length / 2;
        this->_sigma = sigma;
    }
    else {
        this->_kernel_data = kernel_data;
        this->_ker_length = ker_length;
        this->_mean = mean;
        this->_sigma = sigma;
    }
}



void decx::vis::gaussian_kernel1D::generate()
{
    float sum = 0, val;
    for (int i = 0; i < this->_ker_length; ++i) {
        val = exp(powf(i - this->_mean, 2) / (-2.f * powf(this->_sigma, 2)));
        this->_kernel_data.ptr[i] = val;
        sum += val;
    }

    for (int i = 0; i < this->_ker_length; ++i) {
        this->_kernel_data.ptr[i] /= sum;
    }
}


decx::vis::gaussian_kernel1D::~gaussian_kernel1D()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_kernel_data);
}
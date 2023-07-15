/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
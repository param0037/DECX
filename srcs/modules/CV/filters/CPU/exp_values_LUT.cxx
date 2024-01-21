/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "exp_values_LUT.h"
#include "../../../core/allocators.h"


decx::vis::_exp_LUT::_exp_LUT(const size_t length, const float sigma)
{
    this->_len = length;
    if (decx::alloc::_host_virtual_page_malloc(&this->_LUT, this->_len * sizeof(float))) {
        exit(-1);
    }
    for (int i = 0; i < this->_len; ++i) {
        this->_LUT.ptr[i] = expf(-(float)(i * i) / this->_sigma);
    }
}


size_t decx::vis::_exp_LUT::Length()
{
    return this->_len;
}



float* decx::vis::_exp_LUT::get_data()
{
    return this->_LUT.ptr;
}



decx::vis::_exp_LUT::~_exp_LUT()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_LUT);
}
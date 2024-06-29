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


#include "conv_utils.h"


void decx::conv::_matrix_configs::gen_matrix_configs(decx::_Matrix* _host_mat)
{
    this->_height = _host_mat->Height();
    this->_pitch = _host_mat->Pitch();
    this->_width = _host_mat->Width();
    this->_ptr = _host_mat->Mat.ptr;
}


void decx::conv::_matrix_configs::gen_matrix_configs(decx::_GPU_Matrix* _device_mat)
{
    this->_height = _device_mat->Height();
    this->_pitch = _device_mat->Pitch();
    this->_width = _device_mat->Width();
    this->_ptr = _device_mat->Mat.ptr;
}


void decx::conv::_matrix_configs::gen_matrix_configs(decx::_MatrixArray* _host_mat)
{
    this->_height = _host_mat->Height();
    this->_pitch = _host_mat->Pitch();
    this->_width = _host_mat->Width();
    this->_matrix_num = _host_mat->ArrayNumber;
    this->_ptr = _host_mat->MatArr.ptr;
    this->_ptr_array = _host_mat->MatptrArr.ptr;
}



void decx::conv::_matrix_configs::gen_matrix_configs(decx::_GPU_MatrixArray* _host_mat)
{
    this->_height = _host_mat->Height();
    this->_pitch = _host_mat->Pitch();
    this->_width = _host_mat->Width();
    this->_matrix_num = _host_mat->ArrayNumber;
    this->_ptr = _host_mat->MatArr.ptr;
    this->_ptr_array = _host_mat->MatptrArr.ptr;
}



decx::conv::_matrix_configs::_matrix_configs() {}
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
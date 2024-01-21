/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "classes_util.h"
#include "DecxNumber.h"


int32_t de::DecxNumber::Type() const
{
    return this->_data_type_flag;
}



void* de::DecxNumber::get_data_ptr()
{
    return &this->_number._fp64;
}



void de::DecxNumber::set_type_flag(const int32_t _type_flag)
{
    this->_data_type_flag = _type_flag;
}
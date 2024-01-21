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


__host__ __device__
de::complex_h::complex_h(const ushort Freal, const ushort Fimage) {
    this->real = Freal;
    this->image = Fimage;
}


__host__ __device__
de::complex_h::complex_h() {}


__device__
void de::complex_h::dev_construct_with_phase(const __half angle)
{
#if __ABOVE_SM_53
    *((__half*)&this->real) = hcos(angle);
    *((__half*)&this->image) = hsin(angle);
#endif
}



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
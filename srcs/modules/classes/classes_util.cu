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


__device__
void de::complex_f::dev_construct_with_phase(const float angle) {
    this->real = __cosf(angle);
    this->image = __sinf(angle);
}


__host__ __device__
de::complex_f::complex_f(const float Freal, const float Fimage) {
    this->real = Freal;
    this->image = Fimage;
}



__host__ __device__
de::complex_f::complex_f() 
{
    this->real = 0;
    this->image = 0;
}



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
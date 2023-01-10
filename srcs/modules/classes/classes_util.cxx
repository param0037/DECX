/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "classes_util.h"


void de::complex_f::construct_with_phase(const float angle) {
    this->real = cosf(angle);
    this->image = sinf(angle);
}



de::complex_f::complex_f(const float Freal, const float Fimage) {
    real = Freal;
    image = Fimage;
}


de::complex_f::complex_f() {}
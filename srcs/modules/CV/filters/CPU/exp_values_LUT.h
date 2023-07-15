/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _EXP_VALUES_LUT_H_
#define _EXP_VALUES_LUT_H_


#include "../../../core/basic.h"
#include "../../../core/memory_management/PtrInfo.h"


namespace decx
{
    namespace vis {
        class _exp_LUT;
    }
}



class decx::vis::_exp_LUT
{
private:
    decx::PtrInfo<float> _LUT;

public:
    size_t _len;
    float _sigma;

    _exp_LUT(const size_t length, const float sigma);


    size_t Length();


    float* get_data();


    ~_exp_LUT();
};


#endif
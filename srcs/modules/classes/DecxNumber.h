/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DECX_NUMBER_H_
#define _DECX_NUMBER_H_


#include "classes_util.h"
#include "type_info.h"


namespace de
{
    class DecxNumber;
}



class _DECX_API_ de::DecxNumber
{
private:
    union __align__(8) _data
    {
        float _fp32;
        double _fp64 = 0;
        de::CPf _cplf32;
        uint8_t _u8;
        int32_t _i32;
        de::Half _fp16;
    };
    _data _number;
    int32_t _data_type_flag;

public:

    int32_t Type() const;


    void* get_data_ptr();


    void set_type_flag(const int32_t _type_flag);
};


#endif
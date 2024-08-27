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


#ifndef _NUMBER_H_
#define _NUMBER_H_


#include "classes_util.h"
#include "type_info.h"


namespace de
{
    class Number;

    typedef const de::Number& InputNumber;
    typedef de::Number& OutputNumber;
    typedef de::Number& InOutNumber;
}



class _DECX_API_ de::Number
{
private:
    union __align__(16) _data
    {
        float       _fp32;
        double      _fp64;
        de::CPf     _cplxf32;
        de::CPd     _cplxd64;
        uint8_t     _u8;
        int32_t     _i32;
        de::Half    _fp16;
    };
    _data _number;
    int32_t _data_type_flag;

public:

    int32_t Type() const{
        return this->_data_type_flag;
    }

    
    template <typename _data_type>
    _data_type* get_data_ptr() const{
        return ((_data_type*)&this->_number);
    }


    template <typename _data_type>
    _data_type& get_data_ref() const{
        return *((_data_type*)&this->_number);
    }


    void set_type_flag(const de::_DATA_TYPES_FLAGS_ _type_flag){
        this->_data_type_flag = _type_flag;
    }
    
        
    de::Number& operator=(const float& src)
    {
        this->_number._fp32 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP32_;
        return *this;
    }


    de::Number& operator=(const double& src)
    {
        this->_number._fp64 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP64_;
        return *this;
    }


    de::Number& operator=(const de::CPf& src)
    {
        this->_number._cplxf32 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_;
        return *this;
    }


    de::Number& operator=(const de::CPd& src)
    {
        this->_number._cplxd64 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_;
        return *this;
    }


    de::Number& operator=(const uint8_t& src)
    {
        this->_number._u8 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_UINT8_;
        return *this;
    }


    de::Number& operator=(const int32_t& src)
    {
        this->_number._i32 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_INT32_;
        return *this;
    }


    de::Number& operator=(const de::Half& src)
    {
        this->_number._fp16 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP16_;
        return *this;
    }

};


#endif
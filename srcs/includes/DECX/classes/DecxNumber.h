/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#ifndef _DECXNUMBER_H_
#define _DECXNUMBER_H_


#include "../basic_proc/type_cast.h"


namespace de
{
class _DECX_API_ Number
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

		_data(){
			((de::CPd*)this)->real = 0.0;
			((de::CPd*)this)->image = 0.0;
		}
    };
    _data _number;
    int32_t _data_type_flag;

public:
	Number() {
		this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_VOID_;
	}

    Number(const int32_t val) {
        this->_number._i32 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_INT32_;
    }

    Number(const uint8_t val) {
        this->_number._u8 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_UINT8_;
    }

    Number(const float val) {
        this->_number._fp32 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP32_;
    }

    Number(const double val) {
        this->_number._fp64 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP64_;
    }

    Number(const de::Half val) {
        this->_number._fp16 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP16_;
    }

    Number(const de::CPf val) {
        this->_number._cplxf32 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_;
    }

    Number(const de::CPd val) {
        this->_number._cplxd64 = val;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_;
    }


	int32_t Type() const{
		return this->_data_type_flag;
	}


	template <typename _Ty>
	_Ty get_data()
	{
		return *((_Ty*)&this->_number);
	}

    de::Number& operator=(const float src)
    {
        this->_number._fp32 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP32_;
        return *this;
    }


    de::Number& operator=(const double src)
    {
        this->_number._fp64 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP64_;
        return *this;
    }


    de::Number& operator=(const de::CPf src)
    {
        this->_number._cplxf32 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_;
        return *this;
    }


    de::Number& operator=(const de::CPd src)
    {
        this->_number._cplxd64 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_;
        return *this;
    }


    de::Number& operator=(const uint8_t src)
    {
        this->_number._u8 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_UINT8_;
        return *this;
    }


    de::Number& operator=(const int32_t src)
    {
        this->_number._i32 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_INT32_;
        return *this;
    }


    de::Number& operator=(const de::Half src)
    {
        this->_number._fp16 = src;
        this->_data_type_flag = de::_DATA_TYPES_FLAGS_::_FP16_;
        return *this;
    }

	void Print()
	{
		switch (this->_data_type_flag)
		{
		case de::_DATA_TYPES_FLAGS_::_FP32_:
			printf("%f\n", this->get_data<float>());
			break;

		case de::_DATA_TYPES_FLAGS_::_FP16_:
			printf("%f\n", de::Half2Float(this->get_data<de::Half>()));
			break;

		case de::_DATA_TYPES_FLAGS_::_FP64_:
			printf("%f\n", this->get_data<double>());
			break;

		case de::_DATA_TYPES_FLAGS_::_INT32_:
			printf("%d\n", this->get_data<int32_t>());
			break;

		case de::_DATA_TYPES_FLAGS_::_UINT8_:
			printf("%d\n", (int32_t)this->get_data<uint8_t>());
			break;

		case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
			printf("(%f, %f)\n", this->get_data<de::complex_f>().real, this->get_data<de::complex_f>().image);
			break;
		default:
			break;
		}
	}
};
}


#endif
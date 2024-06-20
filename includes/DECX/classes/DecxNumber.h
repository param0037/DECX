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
	class _DECX_API_ DecxNumber
	{
	private:
		double _number;
		int32_t _data_type_flag;

	public:
		DecxNumber() {}

		int32_t Type() const;

		template <typename _Ty>
		_Ty get_data()
		{
			return *((_Ty*)&this->_number);
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
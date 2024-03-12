/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GPU_MATRIX_H_
#define _GPU_MATRIX_H_

#include "../basic.h"

namespace de
{
    class _DECX_API_ GPU_Matrix
    {
    public:
        GPU_Matrix() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::GPU_Matrix& SoftCopy(de::GPU_Matrix& src) = 0;


        virtual de::_DATA_FORMATS_ Format() const = 0;


        ~GPU_Matrix() {}
    };


	_DECX_API_ de::GPU_Matrix& CreateGPUMatrixRef();


	_DECX_API_ de::GPU_Matrix* CreateGPUMatrixPtr();



    _DECX_API_ de::GPU_Matrix* CreateGPUMatrixPtr(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
        const de::_DATA_FORMATS_ format = de::_NA_);


    _DECX_API_ de::GPU_Matrix& CreateGPUMatrixRef(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
        const de::_DATA_FORMATS_ format = de::_NA_);


	namespace cuda
	{
		_DECX_API_ de::DH PinMemory(de::Matrix& src);


		_DECX_API_ de::DH UnpinMemory(de::Matrix& src);
	}
}

#endif

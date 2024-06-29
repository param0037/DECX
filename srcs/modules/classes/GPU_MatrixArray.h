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


#ifndef _GPU_MATRIXARRAY_H_
#define _GPU_MATRIXARRAY_H_

#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/MatrixArray.h"


namespace de
{
    class _DECX_API_ GPU_MatrixArray
    {
    public:
        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint MatrixNumber() const = 0;


        virtual de::GPU_MatrixArray& SoftCopy(de::GPU_MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };
}



namespace decx
{
    class _DECX_API_ _GPU_MatrixArray : public de::GPU_MatrixArray
    {
        // call AFTER attributes are assigned !
        // Once called, the data space will be re-constructed unconditionally, according to the 
        // attributes, the previous data will be lost
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        // Once called, the data space will be constructed unconditionally, according to the 
        // attributes
        void alloc_data_space();


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum);


        decx::_matrix_layout _layout;


        bool _init;

    public:
        decx::PtrInfo<void> MatArr;
        decx::PtrInfo<void*> MatptrArr;

        de::_DATA_TYPES_FLAGS_ type;

        size_t element_num, _element_num,
            total_bytes,    // The real total bytes of the MatrixArray memory block, ATTENTION : NO '_' at the front
            ArrayNumber;    // The number of matrices that share the same sizes

        size_t plane, _plane;


        _GPU_MatrixArray();


        _GPU_MatrixArray(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        void construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        virtual uint32_t Width() const;


        virtual uint32_t Height() const;


        virtual uint32_t MatrixNumber() const;


        virtual de::GPU_MatrixArray& SoftCopy(de::GPU_MatrixArray& src);


        virtual void release();


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        uint32_t Pitch() const;


        const decx::_matrix_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;
    };
}



namespace de
{
    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef();


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr();


    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _Mat_number);


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _Mat_number);
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::MatrixArray& src);


        _DECX_API_ de::DH UnpinMemory(de::MatrixArray& src);
    }
}


#endif
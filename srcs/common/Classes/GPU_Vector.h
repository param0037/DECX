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


#ifndef _GPU_VECTOR_H_
#define _GPU_VECTOR_H_

#include "../basic.h"
#include "../../modules/core/allocators.h"
#include "../../modules/core/cudaStream_management/cudaEvent_queue.h"
#include "../../modules/core/cudaStream_management/cudaStream_queue.h"
#include "Vector.h"

namespace de
{
    class _DECX_API_ GPU_Vector
    {
    public:
        GPU_Vector() {}


        virtual size_t Len() const = 0;


        virtual void release() = 0;


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src) = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        ~GPU_Vector() {}
    };

    typedef const de::GPU_Vector& InputGPUVector;
    typedef de::GPU_Vector& OutputGPUVector;
    typedef de::GPU_Vector& InOutGPUVector;
}




namespace decx
{
    class _DECX_API_ _GPU_Vector : public de::GPU_Vector
    {
    private:
        bool _init;

    public:
        size_t length,
            _length,    // It is aligned with 4
            total_bytes;

        de::_DATA_TYPES_FLAGS_ type;
        uint8_t _single_element_size;


        decx::PtrInfo<void> Vec;


        _GPU_Vector();


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, size_t length);


        void construct(const de::_DATA_TYPES_FLAGS_ _type, const size_t length);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, const size_t length);


        void alloc_data_space();


        void re_alloc_data_space();


        _GPU_Vector(const de::_DATA_TYPES_FLAGS_ _type, size_t length);


        virtual uint64_t Len() const;


        virtual void release();


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src);


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        bool is_init() const;


        uint64_t _Length() const;


        uint64_t get_total_bytes() const;


        ~_GPU_Vector();
    };
}




namespace de
{
    _DECX_API_ de::GPU_Vector& CreateGPUVectorRef();


    _DECX_API_ de::GPU_Vector* CreateGPUVectorPtr();


    _DECX_API_ de::GPU_Vector& CreateGPUVectorRef(const de::_DATA_TYPES_FLAGS_ _type, const size_t length);


    _DECX_API_ de::GPU_Vector* CreateGPUVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, const size_t length);
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::Vector& src);


        _DECX_API_ de::DH UnpinMemory(de::Vector& src);
    }
}




#endif
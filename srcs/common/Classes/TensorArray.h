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


#ifndef _TENSORARRAY_H_
#define _TENSORARRAY_H_


#include "../basic.h"
#include "../../modules/core/allocators.h"
#include "classes_util.h"
#include "Tensor.h"


namespace de
{
    class 
#if _CPP_EXPORT_ENABLED_
        _DECX_API_
#endif 
        TensorArray
    {
    protected:
        _SHADOW_ATTRIBUTE_(void**) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(decx::_tensor_layout) _exp_tensor_dscr;

    public:
        TensorArray() {}


        virtual uint32_t Width() const = 0;


        virtual uint32_t Height() const = 0;


        virtual uint32_t Depth() const = 0;


        virtual uint32_t TensorNum() const = 0;


        /*virtual float* ptr_fp32(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual int* ptr_int32(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual de::Half* ptr_fp16(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual double* ptr_fp64(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual uint8_t* ptr_uint8(const int x, const int y, const int z, const int tensor_id) = 0;*/


        virtual de::TensorArray& SoftCopy(de::TensorArray& src) = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::DH Extract_SoftCopy(const uint32_t index, de::Tensor& dst) const = 0;


        virtual void release() = 0;
    };
}



/**
* The data storage structure is shown below
* tensor_id
*            <-------------------- dp_x_w --------------------->
*             <---------------- width -------------->
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T          
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |          
*    0       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |          
*            ...                                     ...         |    height
*            ...                                     ...         |          
*            ...                                     ...         |          
*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _          
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*    1       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*            ...                                     ...
*            ...                                     ...
*            ...                                     ...
*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*    .
*    .
*    .
*
* Where : the vector along depth-axis
*    <------------ dpitch ----------->
*    <---- pitch ------>
*    [x x x x x x x x x 0 0 0 0 0 0 0]
*/

namespace decx
{
    class _DECX_API_ _TensorArray : public de::TensorArray
    {
    private:
        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _tensor_num);


        void alloc_data_space();


        void re_alloc_data_space();


        decx::_tensor_layout _layout;

        uint32_t tensor_num;

        de::_DATA_TYPES_FLAGS_ type;

        bool _init;

    public:

        // The data pointer
        decx::PtrInfo<void> TensArr;
        // The pointer array for the pointers of each tensor in the TensorArray
        decx::PtrInfo<void*> TensptrArr;

        /*
         * is the number of ACTIVE elements on a xy, xz, yz-plane,
         *  plane[0] : plane-WH
         *  plane[1] : plane-WD
         *  plane[2] : plane-HD
         */
         //uint64_t plane[3];

         // The true size of a Tensor, including pitch
        uint64_t _gap;

        // The number of all the active elements in the TensorArray
        uint64_t element_num;

        // The number of all the elements in the TensorArray, including pitch
        uint64_t _element_num;

        // The size of all the elements in the TensorArray, including pitch
        uint64_t total_bytes;


        _TensorArray();


        _TensorArray(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _tensor_num);


        void construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _tensor_num);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _tensor_num);


        virtual uint32_t Width() const;


        virtual uint32_t Height() const;


        virtual uint32_t Depth() const;


        virtual uint32_t TensorNum() const;


        /*virtual float* ptr_fp32(const int x, const int y, const int z, const int tensor_id);
        virtual int* ptr_int32(const int x, const int y, const int z, const int tensor_id);
        virtual de::Half* ptr_fp16(const int x, const int y, const int z, const int tensor_id);
        virtual double* ptr_fp64(const int x, const int y, const int z, const int tensor_id);
        virtual uint8_t* ptr_uint8(const int x, const int y, const int z, const int tensor_id);*/


        virtual de::TensorArray& SoftCopy(de::TensorArray& src);


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        virtual de::DH Extract_SoftCopy(const uint32_t index, de::Tensor& dst) const;


        virtual void release();


        const decx::_tensor_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;
    };
}


#if _CPP_EXPORT_ENABLED_
namespace de
{
    _DECX_API_ de::TensorArray& CreateTensorArrayRef();


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr();


    _DECX_API_ de::TensorArray& CreateTensorArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t width, const uint32_t height, const uint32_t depth, const uint32_t tensor_num);


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t width, const uint32_t height, const uint32_t depth, const uint32_t tensor_num);
}
#endif


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct decx::_TensorArray* DECX_TensorArray;


    _DECX_API_ DECX_TensorArray DE_CreateEmptyTensorArray();


    _DECX_API_ DECX_TensorArray DE_CreateTensorArray(const int8_t type, const uint32_t _width, const uint32_t _height,
        const uint32_t _depth, const uint32_t tensor_num);
#ifdef __cplusplus
}
#endif      // # ifdef __cplusplus
#endif      // #if _C_EXPORT_ENABLED_


#endif        // #ifndef _TENSORARRAY_H_
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


#ifndef _GPU_TENSOR_H_
#define _GPU_TENSOR_H_

#include "../basic.h"
#include "../../modules/core/allocators.h"
#include "../../modules/core/cudaStream_management/cudaEvent_queue.h"
#include "../../modules/core/cudaStream_management/cudaStream_queue.h"
#include "Tensor.h"


namespace de
{
    class _DECX_API_ GPU_Tensor
    {
    public:
        GPU_Tensor() {}


        virtual uint32_t Width() const = 0;


        virtual uint32_t Height() const = 0;


        virtual uint32_t Depth() const = 0;


        virtual de::GPU_Tensor& SoftCopy(de::GPU_Tensor& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };
}





/**
* The data storage structure is shown below
*
*            <-------------------- dp_x_wp(4x) ------------------>
*            <--------------- width ---------------->
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |
*            ...                                     ...   ...   |    height
*            ...                                     ...   ...   |
*            ...                                     ...   ...   |
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _
*
* Where : the vector along depth-axis
*    <------------ dpitch ----------->
*    <---- pitch ------>
*    [x x x x x x x x x 0 0 0 0 0 0 0]
*/


namespace decx
{
    // z-channel stored adjacently
    class _DECX_API_ _GPU_Tensor : public de::GPU_Tensor
    {
    private:
        
        void alloc_data_space();


        void re_alloc_data_space(decx::cuda_stream* S = NULL);


        de::_DATA_TYPES_FLAGS_ type;

        bool _init;

    public:
        decx::_tensor_layout _layout;


        decx::PtrInfo<void> Tens;
        size_t element_num;        // is the number of all the ACTIVE elements
        size_t total_bytes;        // is the size of ALL(including pitch) elements


        size_t _element_num;        // the total number of elements, including Non_active numbers


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


        void construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth,
            decx::cuda_stream* S = NULL);


        _GPU_Tensor();


        _GPU_Tensor(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);



        virtual uint32_t Width() const;


        virtual uint32_t Height() const;


        virtual uint32_t Depth() const;


        virtual de::GPU_Tensor& SoftCopy(de::GPU_Tensor& src);



        virtual void release();


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        const decx::_tensor_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;
    };
}



namespace de
{
    _DECX_API_ de::GPU_Tensor* CreateGPUTensorPtr();


    _DECX_API_ de::GPU_Tensor& CreateGPUTensorRef();


    _DECX_API_ de::GPU_Tensor* CreateGPUTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


    _DECX_API_ de::GPU_Tensor& CreateGPUTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::Tensor& src);


        _DECX_API_ de::DH UnpinMemory(de::Tensor& src);
    }
}



#endif      // #ifndef _GPU_TENSOR_H_
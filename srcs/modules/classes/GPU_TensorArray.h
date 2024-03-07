/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GPU_TENSORARRAY_H_
#define _GPU_TENSORARRAY_H_

#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/classes_util.h"
#include "TensorArray.h"
#include "GPU_Tensor.h"


namespace de
{
    class _DECX_API_ GPU_TensorArray
    {
    public:
        GPU_TensorArray() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual uint TensorNum() const = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::GPU_TensorArray& SoftCopy(const de::GPU_TensorArray& src) = 0;


        virtual de::DH Extract_SoftCopy(const uint32_t index, de::GPU_Tensor& dst) const = 0;


        virtual void release() = 0;
    };
}



/**
* The data storage structure is shown below
* tensor_id
*            <-------------------- dp_x_w --------------------->
*             <---------------- width -------------->
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T            T
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            |
*    0       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            |
*            ...                                     ...         |    height  |    hpitch(2x)
*            ...                                     ...         |            |
*            ...                                     ...         |            |
*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _            _
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
    
    class _DECX_API_ _GPU_TensorArray : public de::GPU_TensorArray
    {
    private:
        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num);


        void alloc_data_space();


        void re_alloc_data_space();

        decx::_tensor_layout _layout;

        de::_DATA_TYPES_FLAGS_ type;

        uint32_t tensor_num;

        bool _init;

    public:
        
        // The data pointer
        decx::PtrInfo<void> TensArr;
        // The pointer array for the pointers of each tensor in the TensorArray
        decx::PtrInfo<void*> TensptrArr;


        // The true size of a Tensor, including pitch
        size_t _gap;

        // The number of all the active elements in the TensorArray
        size_t element_num;

        // The number of all the elements in the TensorArray, including pitch
        size_t _element_num;

        // The size of all the elements in the TensorArray, including pitch
        size_t total_bytes;


        void construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num);


        _GPU_TensorArray();


        _GPU_TensorArray(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num);


        const decx::_tensor_layout& get_layout() const;


        virtual uint Width() const;


        virtual uint Height() const;


        virtual uint Depth() const;


        virtual uint TensorNum() const;


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        virtual de::GPU_TensorArray& SoftCopy(const de::GPU_TensorArray& src);


        virtual de::DH Extract_SoftCopy(const uint32_t index, de::GPU_Tensor& dst) const;


        virtual void release();


        bool is_init() const;


        uint64_t get_total_bytes() const;
    };
}



namespace de
{
    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef();



    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr();



    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num);



    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num);
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::TensorArray& src);


        _DECX_API_ de::DH UnpinMemory(de::TensorArray& src);
    }
}


#endif
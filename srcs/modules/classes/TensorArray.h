/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _TENSORARRAY_H_
#define _TENSORARRAY_H_


#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/classes_util.h"
#include "store_types.h"
#include "Tensor.h"


namespace de
{
    class _DECX_API_ TensorArray
    {
    public:
        TensorArray() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual uint TensorNum() = 0;


        virtual void* index(const int x, const int y, const int z, const int tensor_id) = 0;


        virtual de::TensorArray& SoftCopy(de::TensorArray& src) = 0;


        virtual int Type() = 0;


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
    class _DECX_API_ _TensorArray : public de::TensorArray
    {
    private:
        void _attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type);


        void alloc_data_space();


        void re_alloc_data_space();

    public:
        uint width,
             height,
             depth,
             tensor_num;

        int _store_type;

        // The data pointer
        decx::PtrInfo<void> TensArr;
        // The pointer array for the pointers of each tensor in the TensorArray
        decx::PtrInfo<void*> TensptrArr;

        uint dpitch;            // NOT IN BYTES, the true depth (4x)
        uint wpitch;            // NOT IN BYTES, the true width (2x)
        size_t dp_x_wp;            // NOT IN BYTES, true depth multiply true width

        int type, _single_element_size;

        /*
         * is the number of ACTIVE elements on a xy, xz, yz-plane,
         *  plane[0] : plane-WH
         *  plane[1] : plane-WD
         *  plane[2] : plane-HD
         */
        size_t plane[3];

        // The true size of a Tensor, including pitch
        size_t _gap;

        // The number of all the active elements in the TensorArray
        size_t element_num;

        // The number of all the elements in the TensorArray, including pitch
        size_t _element_num;

        // The size of all the elements in the TensorArray, including pitch
        size_t total_bytes;


        _TensorArray();


        _TensorArray(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type);


        void construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int flag);


        void re_construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int flag);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint Depth() { return this->depth; }


        virtual uint TensorNum() { return this->tensor_num; }


        virtual void* index(const int x, const int y, const int z, const int tensor_id);


        virtual de::TensorArray& SoftCopy(de::TensorArray& src);


        virtual int Type();


        virtual void release();
    };
}


#endif        // #ifndef _TENSORARRAY_H_
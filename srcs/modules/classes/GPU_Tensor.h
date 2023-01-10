/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GPU_TENSOR_H_
#define _GPU_TENSOR_H_

#include "../core/basic.h"
#include "Tensor.h"


namespace de
{
    class _DECX_API_ GPU_Tensor
    {
    public:
        GPU_Tensor() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual de::GPU_Tensor& SoftCopy(de::GPU_Tensor& src) = 0;


        virtual void Load_from_host(de::Tensor& src) = 0;


        virtual void Load_to_host(de::Tensor& dst) = 0;


        virtual void release() = 0;


        virtual int Type() = 0;
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
        void _attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth);


        void alloc_data_space();


        void re_alloc_data_space();

    public:
        uint width,
            height,
            depth;

        uint dpitch;            // NOT IN BYTES, the true depth (4x)
        uint wpitch;            // NOT IN BYTES, the true width (4x)
        size_t dp_x_wp;            // NOT IN BYTES, true depth multiply true width

        int type, _single_element_size;

        decx::PtrInfo<void> Tens;
        size_t element_num;        // is the number of all the ACTIVE elements
        size_t total_bytes;        // is the size of ALL(including pitch) elements

        /*
         * is the number of ACTIVE elements on a xy, xz, yz-plane,
         *  plane[0] : plane-WH
         *  plane[1] : plane-WD
         *  plane[2] : plane-HD
         */
        size_t plane[3];


        size_t _element_num;        // the total number of elements, including Non_active numbers


        void construct(const int _type, const uint _width, const uint _height, const uint _depth);


        void re_construct(const int _type, const uint _width, const uint _height, const uint _depth);


        _GPU_Tensor();


        _GPU_Tensor(const int _type, const uint _width, const uint _height, const uint _depth);



        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint Depth() { return this->depth; }


        virtual de::GPU_Tensor& SoftCopy(de::GPU_Tensor& src);


        virtual void Load_from_host(de::Tensor& src);


        virtual void Load_to_host(de::Tensor& dst);


        virtual void release();


        virtual int Type();
    };
}


#endif      // #ifndef _GPU_TENSOR_H_
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
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
        uint ArrayNumber();


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint MatrixNumber() = 0;
        

        virtual void Load_from_host(de::MatrixArray &src) = 0;


        virtual void Load_to_host(de::MatrixArray& dst) = 0;


        virtual de::GPU_MatrixArray& SoftCopy(de::GPU_MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual int Type() = 0;
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


        void _attribute_assign(const int _type, uint width, uint height, uint MatrixNum);

    public:
        decx::PtrInfo<void> MatArr;
        decx::PtrInfo<void*> MatptrArr;

        uint width, height;

        int type, _single_element_size;

        size_t element_num, _element_num,
            total_bytes,    // The real total bytes of the MatrixArray memory block, ATTENTION : NO '_' at the front
            ArrayNumber;    // The number of matrices that share the same sizes

        size_t plane, _plane;

        uint pitch,            // the true width (NOT IN BYTES)
            _height;        // the true height

        _GPU_MatrixArray();


        _GPU_MatrixArray(const int _type, uint width, uint height, uint MatrixNum);


        void construct(const int _type, uint width, uint height, uint MatrixNum);


        void re_construct(const int _type, uint width, uint height, uint MatrixNum);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint MatrixNumber() { return this->ArrayNumber; }


        virtual void Load_from_host(de::MatrixArray& src);


        virtual void Load_to_host(de::MatrixArray& dst);


        virtual de::GPU_MatrixArray& SoftCopy(de::GPU_MatrixArray& src);


        virtual void release();


        virtual int Type();
    };
}





#endif
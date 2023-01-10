/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _MATRIXARRAY_H_
#define _MATRIXARRAY_H_


#include "../core/basic.h"
#include "Matrix.h"
#include "../core/allocators.h"
#include "store_types.h"
#include "type_info.h"


namespace de
{
    /*
    * This class is for matrices array, the matrices included share the same sizes, and store
    * one by one in the memory block, without gap; Compared with de::Tensor<T>, the channel "z"
    * is separated.
    */
    class _DECX_API_ MatrixArray
    {
    public:
        uint ArrayNumber;        


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint MatrixNumber() = 0;


        virtual float*      ptr_fp32(const uint row, const uint col, const uint _seq) = 0;
        virtual double*     ptr_fp64(const uint row, const uint col, const uint _seq) = 0;
        virtual int*        ptr_int32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::CPf*    ptr_cpl32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::Half*   ptr_fp16(const uint row, const uint col, const uint _seq) = 0;


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual int Type() = 0;
    };
}




/**
* DATA STORAGE PATTERN:
* |        pitch       |
* |     width      |
*  ____________________ __
* |xxxxxxxxxxxxxxxxx   |
* |xxxxxxxxxxxxxxxxx   | height
* |xxxxxxxxxxxxxxxxx   |
* |！！！！！！！！！！！！！！！！！！！！|！！
* |xxxxxxxxxxxxxxxxx   |
* |xxxxxxxxxxxxxxxxx   | height
* |xxxxxxxxxxxxxxxxx   |
*  ！！！！！！！！！！！！！！！！！！！！ ！！
*/

namespace decx
{
    class _DECX_API_ _MatrixArray : public de::MatrixArray
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(const int _type, uint width, uint height, uint MatrixNum, const int flag);

    public:
        int _store_type;

        decx::PtrInfo<void> MatArr;
        decx::PtrInfo<void*> MatptrArr;

        uint width, height;

        size_t element_num, _element_num,
            total_bytes,    // The real total bytes of the MatrixArray memory block, ATTENTION : NO '_' at the front
            ArrayNumber;    // The number of matrices that share the same sizes

        size_t plane, _plane;

        int type, _single_element_size;

        uint pitch,            // the true width (NOT IN BYTES)
            _height;        // the true height

        void construct(const int _type, uint width, uint height, uint MatrixNum, const int flag);


        void re_construct(const int _type, uint width, uint height, uint MatrixNum, const int flag);


        _MatrixArray();


        _MatrixArray(const int _type, uint width, uint height, uint MatrixNum, const int flag);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint MatrixNumber() { return this->ArrayNumber; }


        virtual float*      ptr_fp32(const uint row, const uint col, const uint _seq);
        virtual double*     ptr_fp64(const uint row, const uint col, const uint _seq);
        virtual int*        ptr_int32(const uint row, const uint col, const uint _seq);
        virtual de::CPf*    ptr_cpl32(const uint row, const uint col, const uint _seq);
        virtual de::Half*   ptr_fp16(const uint row, const uint col, const uint _seq);


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src);


        virtual void release();


        virtual int Type();
    };

}




#endif        // #ifndef _MATRIXARRAY_H_
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIXARRAY_H_
#define _MATRIXARRAY_H_


#include "../core/basic.h"
#include "Matrix.h"
#include "../core/allocators.h"
#include "type_info.h"


namespace de
{
    /*
    * This class is for matrices array, the matrices included share the same sizes, and store
    * one by one in the memory block, without gap; Compared with de::Tensor<T>, the channel "z"
    * is separated.
    */
    class 
#if _CPP_EXPORT_ENABLED_
        _DECX_API_
#endif 
        MatrixArray
    {
    public:
        uint ArrayNumber;        


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint MatrixNumber() const = 0;


        virtual float*      ptr_fp32(const uint row, const uint col, const uint _seq) = 0;
        virtual double*     ptr_fp64(const uint row, const uint col, const uint _seq) = 0;
        virtual int*        ptr_int32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::CPf*    ptr_cpl32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::Half*   ptr_fp16(const uint row, const uint col, const uint _seq) = 0;
        virtual uint8_t*    ptr_uint8(const uint row, const uint col, const uint _seq) = 0;


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
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


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum);


        decx::_matrix_layout _layout;

        bool _init;

    public:
        
        decx::PtrInfo<void> MatArr;
        decx::PtrInfo<void*> MatptrArr;

        //uint width, height;

        size_t element_num, _element_num,
            total_bytes,    // The real total bytes of the MatrixArray memory block, ATTENTION : NO '_' at the front
            ArrayNumber;    // The number of matrices that share the same sizes

        size_t plane, _plane;

        de::_DATA_TYPES_FLAGS_ type;
        uint8_t _single_element_size;

        //uint pitch,            // the true width (NOT IN BYTES)
        //    _height;        // the true height

        void construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        _MatrixArray();


        _MatrixArray(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        virtual uint32_t Width() const;


        virtual uint32_t Height() const;


        virtual uint32_t MatrixNumber() const;


        virtual float*      ptr_fp32(const uint row, const uint col, const uint _seq);
        virtual double*     ptr_fp64(const uint row, const uint col, const uint _seq);
        virtual int*        ptr_int32(const uint row, const uint col, const uint _seq);
        virtual de::CPf*    ptr_cpl32(const uint row, const uint col, const uint _seq);
        virtual de::Half*   ptr_fp16(const uint row, const uint col, const uint _seq);
        virtual uint8_t*    ptr_uint8(const uint row, const uint col, const uint _seq);


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src);


        virtual void release();


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);
        

        uint32_t Pitch() const;


        uint32_t Array_num() const;


        const decx::_matrix_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;
    };

}


#if _CPP_EXPORT_ENABLED_
namespace de
{
    _DECX_API_ de::MatrixArray& CreateMatrixArrayRef();


    _DECX_API_ de::MatrixArray* CreateMatrixArrayPtr();


    _DECX_API_ de::MatrixArray& CreateMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


    _DECX_API_ de::MatrixArray* CreateMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);
}
#endif


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct decx::_MatrixArray* DECX_MatrixArray;


    _DECX_API_ DECX_MatrixArray DE_CreateEmptyMatrixArray();


    _DECX_API_ DECX_MatrixArray DE_CreateMatrixArray(const int8_t type, const uint32_t _width, const uint32_t _height,
        uint32_t MatrixNum);


    // _DECX_API_ DECX_Handle DE_GetMatrixArrayProp(const DECX_MatrixArray src, DECX_MatrixLayout* prop);
#ifdef __cplusplus
}
#endif          // #ifdef __cplusplus
#endif          // #if _C_EXPORT_ENABLED_


#endif        // #ifndef _MATRIXARRAY_H_
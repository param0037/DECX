/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#ifndef _MATRIXARRAY_H_
#define _MATRIXARRAY_H_

#include "../basic.h"

#ifdef __cplusplus
namespace de
{
	/*
	* This class is for matrices array, the matrices included share the same sizes, and store
	* one by one in the memory block, without gap; Compared with de::Tensor<T>, the channel "z"
	* is separated.
	*/
    class _DECX_API_ MatrixArray
    {
    protected:
        _SHADOW_ATTRIBUTE_(void**) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(de::MatrixLayout) _exp_matrix_dscr;

    public:
        uint ArrayNumber;


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint MatrixNumber() const = 0;


        /*virtual float* ptr_fp32(const uint row, const uint col, const uint _seq) = 0;
        virtual double* ptr_fp64(const uint row, const uint col, const uint _seq) = 0;
        virtual int* ptr_int32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::CPf* ptr_cpl32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::Half* ptr_fp16(const uint row, const uint col, const uint _seq) = 0;
        virtual uint8_t* ptr_uint8(const uint row, const uint col, const uint _seq) = 0;*/

        template <typename _ptr_type>
        _ptr_type* ptr(const uint row, const uint col, const uint _seq)
        {
            return ((_ptr_type*)(*this->_exp_data_ptr)[_seq]) + this->_exp_matrix_dscr->pitch * row + col;
        }


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };


	_DECX_API_
    de::MatrixArray& CreateMatrixArrayRef();

    _DECX_API_
    de::MatrixArray* CreateMatrixArrayPtr();

    _DECX_API_
    de::MatrixArray& CreateMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum);

    _DECX_API_
    de::MatrixArray* CreateMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum);
}
#endif

#ifdef _C_CONTEXT_
typedef struct DECX_MatrixArray_t
{
    void* _segment;
}DECX_MatrixArray;


_DECX_API_ DECX_MatrixArray DE_CreateEmptyMatrixArray();


_DECX_API_ DECX_MatrixArray DE_CreateMatrixArray(const int8_t type, const uint32_t _width, const uint32_t _height,
    uint32_t MatrixNum);
#endif


#endif

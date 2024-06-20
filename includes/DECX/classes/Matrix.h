/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "../basic.h"
#include "class_utils.h"


#ifdef __cplusplus
/**
* in host, allocate page-locaked memory in 8-times both on width and height
* ensure the utilization of __m128 and __m256, as well as multi threads
*/

namespace de
{
    struct MatrixLayout;
}
#endif


#ifdef __cplusplus
struct de::MatrixLayout
#else
struct MatrixLayout
#endif
{
    uint32_t width, height;
    uint32_t pitch;

    int32_t _single_element_size;
};

#ifdef __cplusplus
namespace de 
{
    class _DECX_API_ Matrix
    {
    protected:
        _SHADOW_ATTRIBUTE_(void*) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(de::MatrixLayout) _exp_matrix_dscr;

    public:
        Matrix() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;



        /* return the reference of the element in the matrix, which locates on specific row and colume
        * \params row -> where the element locates on row
        * \params col -> where the element locates on colume
        */
        /*virtual float* ptr_fp32(const int row, const int col) = 0;
        virtual double* ptr_fp64(const int row, const int col) = 0;
        virtual int* ptr_int32(const int row, const int col) = 0;
        virtual de::CPf* ptr_cpl32(const int row, const int col) = 0;
        virtual de::CPd* ptr_cpl64(const int row, const int col) = 0;
        virtual de::Half* ptr_fp16(const int row, const int col) = 0;
        virtual uint8_t* ptr_uint8(const int row, const int col) = 0;*/

        template <typename _ptr_type>
        _ptr_type* ptr(const int row, const int col) 
        {
            return ((_ptr_type*)*this->_exp_data_ptr) + this->_exp_matrix_dscr->pitch * row + col;
        }


        virtual void release() = 0;


        virtual de::Matrix& SoftCopy(de::Matrix& src) = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::_DATA_FORMATS_ Format() const = 0;


        ~Matrix() {}
    };


    _DECX_API_ de::Matrix* CreateMatrixPtr();


    _DECX_API_ de::Matrix& CreateMatrixRef();


    _DECX_API_ de::Matrix* CreateMatrixPtr(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
        const de::_DATA_FORMATS_ format = de::_NA_);


    _DECX_API_ de::Matrix& CreateMatrixRef(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
        const de::_DATA_FORMATS_ format = de::_NA_);
}
#endif



#ifdef _C_CONTEXT_
typedef struct DECX_MatrixLayout_t
{
    uint32_t _width, _height;
    uint32_t _pitch;

    int32_t _single_element_size;
}DECX_MatrixLayout;


typedef struct DECX_Matrix_t* DECX_Matrix;


_DECX_API_ DECX_Matrix DE_CreateEmptyMatrix();


_DECX_API_ DECX_Matrix DE_CreateMatrix(const int8_t type, const uint32_t _width, const uint32_t _height,
    const int8_t format);

/*
* @return : [Width, Height]
*/
_DECX_API_ DECX_Handle DE_GetMatrixProp(const DECX_Matrix src, DECX_MatrixLayout* prop);
#endif


#endif

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


#ifndef _MATRIX_H_
#define _MATRIX_H_


#include "../basic.h"
#include "../../modules/core/allocators.h"
#include "classes_util.h"
#ifdef _DECX_CPU_PARTS_
//#include "../core/thread_management/thread_pool.h"
#endif
#include "type_info.h"

// x86-64, avx256, 256bit alignment
#ifdef __x86_64__
#define _MATRIX_ALIGN_4B_ 8
#define _MATRIX_ALIGN_2B_ 16
#define _MATRIX_ALIGN_8B_ 4
#define _MATRIX_ALIGN_16B_ 2
#define _MATRIX_ALIGN_1B_ 32
#endif

// arm64 neon, 128bit alignment
#ifdef __aarch64__
#define _MATRIX_ALIGN_4B_ 4
#define _MATRIX_ALIGN_2B_ 8
#define _MATRIX_ALIGN_8B_ 2
#define _MATRIX_ALIGN_16B_ 1
#define _MATRIX_ALIGN_1B_ 16
#endif


/**
* in host, allocate page-locaked memory in 8-times both on width and height
* ensure the utilization of __m128 and __m256, as well as multi threads
*/
#ifdef __cplusplus


namespace decx
{
    class _matrix_layout;
}


class decx::_matrix_layout
{
public:
    uint32_t width, height;
    uint32_t pitch;

    int32_t _single_element_size;


    _matrix_layout() {}


    const _matrix_layout& operator=(const _matrix_layout& _src)
    {
        this->width = _src.width;
        this->height = _src.height;

        this->pitch = _src.pitch;
        this->_single_element_size = _src._single_element_size;

        return *this;
    }


    void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height);
};



namespace de {
    class 
#if _CPP_EXPORT_ENABLED_
        _DECX_API_
#endif
        Matrix
    {
    protected:
        _SHADOW_ATTRIBUTE_(void*) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(decx::_matrix_layout) _matrix_dscr;

    public:
        Matrix() {}


        virtual uint32_t Width() const = 0;


        virtual uint32_t Height() const = 0;



        /* return the reference of the element in the matrix, which locates on specific row and colume
        * \params row -> where the element locates on row
        * \params col -> where the element locates on colume
        */
        /*virtual float*      ptr_fp32(const int row, const int col)  = 0;
        virtual double*     ptr_fp64(const int row, const int col)  = 0;
        virtual int*        ptr_int32(const int row, const int col) = 0;
        virtual de::CPf*    ptr_cpl32(const int row, const int col) = 0;
        virtual de::CPd*    ptr_cpl64(const int row, const int col) = 0;
        virtual de::Half*   ptr_fp16(const int row, const int col)  = 0;
        virtual uint8_t*    ptr_uint8(const int row, const int col) = 0;*/
        
        virtual void release() = 0;


        virtual de::Matrix& SoftCopy(de::Matrix& src) = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::_DATA_FORMATS_ Format() const = 0;


        ~Matrix() {}
    };
}


/**
* The data storage structure is shown below:
* 
*            <-------------- width ------------->
*            <---------------- pitch ------------------>
*            [[x x x x x x... ...    ... ...x x x....] T        
*            [[x x x x x x... ...    ... ...x x x....] |    
*            [[x x x x x x... ...    ... ...x x x....] |    
*            ...                                  ...  |    height    
*            ...                                  ...  |        
*            ...                                  ...  |        
*            [[x x x x x x... ...    ... ...x x x....] _
*/


namespace decx
{
    class _DECX_API_ _Matrix : public de::Matrix
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height);

        decx::_matrix_layout _layout;

        //unsigned short Store_Type;

        de::_DATA_TYPES_FLAGS_ type;

        /**
        * Active when the matrix represents image, or contains complex numbers.
        * Otherwise, preserved.
        */
        de::_DATA_FORMATS_ _format;

        bool _init;

        uint64_t    // true_width * true_height
            total_bytes;       // true_width * true_height * sizeof(DATA)

    public:
        
        decx::PtrInfo<void> Mat;


        void construct(const de::_DATA_TYPES_FLAGS_ type, uint32_t width, uint32_t height, const de::_DATA_FORMATS_ format = de::_NA_);


        void re_construct(const de::_DATA_TYPES_FLAGS_ type, uint32_t width, uint32_t height);


        _Matrix();


        _Matrix(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height, const de::_DATA_FORMATS_ format = de::_NA_);


        virtual uint32_t Width() const;


        virtual uint32_t Height() const;



        /*virtual float*      ptr_fp32(const int row, const int col);
        virtual double*     ptr_fp64(const int row, const int col);
        virtual int*        ptr_int32(const int row, const int col);
        virtual de::CPf*    ptr_cpl32(const int row, const int col);
        virtual de::CPd*    ptr_cpl64(const int row, const int col);
        virtual de::Half*   ptr_fp16(const int row, const int col);
        virtual uint8_t*    ptr_uint8(const int row, const int col);*/


        virtual void release();


        virtual de::Matrix& SoftCopy(de::Matrix &src);


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        virtual de::_DATA_FORMATS_ Format() const;


        virtual ~_Matrix();


        uint32_t Pitch() const;


        const decx::_matrix_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;


        de::_DATA_FORMATS_ get_data_format() const;


        void set_data_format(const de::_DATA_FORMATS_& format);
    };
}


#if _CPP_EXPORT_ENABLED_
namespace de
{
    _DECX_API_ de::Matrix* CreateMatrixPtr();


    _DECX_API_ de::Matrix& CreateMatrixRef();


    _DECX_API_ de::Matrix* CreateMatrixPtr(const de::_DATA_TYPES_FLAGS_ type, const uint _width, const uint _height,
        const de::_DATA_FORMATS_ format = de::_NA_);


    _DECX_API_ de::Matrix& CreateMatrixRef(const de::_DATA_TYPES_FLAGS_ type, const uint _width, const uint _height,
        const de::_DATA_FORMATS_ format = de::_NA_);
}
#endif  // #if _CPP_EXPORT_ENABLED_

#endif


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef decx::_Matrix* DECX_Matrix;


    typedef struct DECX_MatrixLayout_t
    {
        uint32_t _width, _height;
        uint32_t _pitch;

        int32_t _single_element_size;
    }DECX_MatrixLayout;


    _DECX_API_ DECX_Matrix DE_CreateEmptyMatrix();


    _DECX_API_ DECX_Matrix DE_CreateMatrix(const int8_t type, const uint32_t _width, const uint32_t _height,
        const int8_t format);

    /*
    * @return : [Width, Height]
    */
    _DECX_API_ DECX_Handle DE_GetMatrixProp(const DECX_Matrix src, DECX_MatrixLayout* prop);
#ifdef __cplusplus
}
#endif      // # ifdef __cplusplus
#endif      // #if _C_EXPORT_ENABLED_


#endif
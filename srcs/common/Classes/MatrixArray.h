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


#ifndef _MATRIXARRAY_H_
#define _MATRIXARRAY_H_


#include "../basic.h"
#include "../../modules/core/allocators.h"
#include "classes_util.h"
#include "type_info.h"
#include "Matrix.h"
#include "../../common/Array/Dynamic_Array.h"

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
    protected:
        // Expose data pointer and matrix layout descriptor to users.
        _SHADOW_ATTRIBUTE_(void**) _exp_data_ptr;

    public:
        uint32_t ArrayNumber;


        virtual uint32_t Width(const uint32_t matrix_id) const = 0;


        virtual uint32_t Height(const uint32_t matrix_id) const = 0;


        virtual uint32_t MatrixNumber() const = 0;


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
* |����������������������������������������|����
* |xxxxxxxxxxxxxxxxx   |
* |xxxxxxxxxxxxxxxxx   | height
* |xxxxxxxxxxxxxxxxx   |
*  ���������������������������������������� ����
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


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        decx::utils::Dynamic_Array<decx::_matrix_layout> _layouts;

        bool _init;
        uint8_t _single_element_size;

        de::_DATA_TYPES_FLAGS_ type;

        uint32_t _matrix_number;

    public:
        
        // decx::PtrInfo<void> MatArr;
        decx::utils::Dynamic_Array<decx::PtrInfo<void>> MatptrArr;

        void construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, const uint32_t matrix_id);


        _MatrixArray();


        _MatrixArray(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, uint32_t MatrixNum);


        virtual uint32_t Width(const uint32_t matrix_id) const;


        virtual uint32_t Height(const uint32_t matrix_id) const;


        virtual uint32_t MatrixNumber() const;


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src);


        virtual void release();


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);
        

        uint32_t Pitch(const uint32_t matrix_id) const;


        uint32_t Array_num() const;


        const decx::_matrix_layout& get_layout(const uint32_t matrix_id) const;


        bool is_init() const;
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
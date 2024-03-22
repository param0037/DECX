/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GPU_MATRIX_H_
#define _GPU_MATRIX_H_

#include "../core/basic.h"
#include "../core/allocators.h"
#include "type_info.h"
#include "Matrix.h"


namespace de
{
    class
#if _CPP_EXPORT_ENABLED_
        _DECX_API_ 
#endif
        GPU_Matrix
    {
    public:
        GPU_Matrix() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;

        
        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::GPU_Matrix& SoftCopy(de::GPU_Matrix& src) = 0;


        virtual de::_DATA_FORMATS_ Format() const = 0;


        ~GPU_Matrix() {}
    };
}



/**
* The data storage structure is shown below
*            <-------------- width ------------->
*            <---------------- pitch ------------------>
*             <-dpitch->
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
    class _DECX_API_ _GPU_Matrix : public de::GPU_Matrix
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height);

        decx::_matrix_layout _layout;

        //unsigned short Store_Type;
        de::_DATA_TYPES_FLAGS_ type;

        de::_DATA_FORMATS_ _format;

        bool _init;

        uint64_t total_bytes;

    public:
        
        decx::PtrInfo<void> Mat;


        void construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height, 
            const de::_DATA_FORMATS_ format = de::_NA_);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t width, uint32_t height);


        _GPU_Matrix();


        _GPU_Matrix(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height,
            const de::_DATA_FORMATS_ format = de::_NA_);


        virtual uint Width() const;


        virtual uint Height() const;


        virtual void release();


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        virtual de::GPU_Matrix& SoftCopy(de::GPU_Matrix& src);


        de::_DATA_FORMATS_ Format() const;


        ~_GPU_Matrix() {}


        uint32_t Pitch();


        const decx::_matrix_layout& get_layout();


        bool is_init();


        uint64_t get_total_bytes();


        de::_DATA_FORMATS_ get_data_format() const;


        void set_data_format(const de::_DATA_FORMATS_& format);
    };
}



namespace de
{
#if _CPP_EXPORT_ENABLED_
    _DECX_API_
#endif 
    de::GPU_Matrix& CreateGPUMatrixRef();


#if _CPP_EXPORT_ENABLED_
    _DECX_API_
#endif 
    de::GPU_Matrix* CreateGPUMatrixPtr();


#if _CPP_EXPORT_ENABLED_
    _DECX_API_
#endif  
    de::GPU_Matrix& CreateGPUMatrixRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t width, const uint32_t height,
        const de::_DATA_FORMATS_ format);


#if _CPP_EXPORT_ENABLED_
    _DECX_API_
#endif  
    de::GPU_Matrix* CreateGPUMatrixPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t width, const uint32_t height,
        const de::_DATA_FORMATS_ format);


    namespace cuda
    {
#if _CPP_EXPORT_ENABLED_
        _DECX_API_
#endif 
        de::DH PinMemory(de::Matrix& src);


#if _CPP_EXPORT_ENABLED_
            _DECX_API_
#endif 
        de::DH UnpinMemory(de::Matrix& src);
    }
}


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
    typedef struct DECX_GPU_Matrix_t
    {
        void* _segment;
    }DECX_GPU_Matrix;


    _DECX_API_ DECX_GPU_Matrix CreateEmptyGPUMatrix();


    _DECX_API_ DECX_GPU_Matrix CreateGPUMatrix(const int8_t type, const uint32_t width, const uint32_t height, 
        const int8_t format);
#ifdef __cplusplus
}
#endif
#endif


#endif

#endif
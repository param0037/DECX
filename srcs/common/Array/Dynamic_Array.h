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

#ifndef _DYNAMIC_ARRAY_H_
#define _DYNAMIC_ARRAY_H_


#include "../basic.h"
#include "../../modules/core/allocators.h"


#define Array_Initial_Length 64
#define Array_Expansion_Length 512


namespace decx
{
    namespace utils {
        template <typename _Ty>
        class Dynamic_Array;
    }
}


template <typename _Ty>
class decx::utils::Dynamic_Array
{
private:
    decx::PtrInfo<_Ty> _buffer;

    decx::PtrInfo<_Ty> _data;

    uint64_t _memory_capacity;
    uint64_t _current_length;
    uint64_t _void_space_from_start;

    // The first element
    _Ty* _begin_ptr;
    // After the last element, not accessible !
    _Ty* _end_ptr;

    /*
    * Eliminates _void_space_from_start by moving the data segment all the way
    * to the beggining of the physical allocated space.
    */
    void move_ahead();


    void memory_expansion();

    /**
     * @brief Judge if _end_ptr will exceed the end of the physical memory
     *
     * @return true if will not exceed (safe)
     * @return false if will exceed (Not safe)
     */
    bool check_vaild_space_req();

public:
    Dynamic_Array();

    /**
     * @brief Get the epscified element of the array by the index
     * @param _index : The index of the element
     * @return _Ty* pointer of the epscified element. If there is no element in the array
     * the return value will be unexpected
     */
    _Ty* operator[](const size_t _index);

    /**
     * @brief Insert element on the back of the array (placement new)
     *
     * @return None
     */
    template<typename... Args>
    void emplace_back(Args&&... args);


    /**
     * @brief Get the last element of the array
     *
     * @return _Ty* pointer of the last element. If there is no element in the array
     * the return value will be unexpected
     */
    _Ty* back();


    /**
     * @brief Delete the element at <index> th position
     *
     * @return void, return none
     */
    void del(const uint64_t index);


    /**
     * @brief Get the first element of the array
     *
     * @return _Ty* pointer of the first element. If there is no element in the array
     * the return value will be unexpected
     */
    _Ty* front();

    /**
     * @brief Get the last element of the array then delete it
     *
     * @return _Ty* pointer of the last element. If there is no element in the array
     * the return value will be unexpected
     */
    _Ty* pop_back();

    /**
     * @brief Get the first element of the array then delete it
     *
     * @return _Ty* pointer of the first element. If there is no element in the array
     * the return value will be unexpected
     */
    _Ty* pop_front();

    /**
     * @brief Get the size of the array
     *
     * @return The size of the array
     */
    uint64_t size() const;


    /**
     * @brief Clear all the element and move Array::_begin_ptr to Array::_data - 1
     *
     */
    void clear();


    ~Dynamic_Array();
};


template <typename _Ty>
decx::utils::Dynamic_Array<_Ty>::Dynamic_Array()
{
    this->_void_space_from_start = 0;

    if (decx::alloc::_host_virtual_page_malloc(&this->_data, Array_Initial_Length * sizeof(_Ty))) {
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&this->_buffer, Array_Initial_Length * sizeof(_Ty))) {
        return;
    }

    this->_memory_capacity = Array_Initial_Length;
    this->_current_length = 0;

    this->_begin_ptr = this->_data.ptr;
    this->_end_ptr = this->_data.ptr;
}


template<typename _Ty>
bool decx::utils::Dynamic_Array<_Ty>::check_vaild_space_req()
{
    return (this->_memory_capacity > (this->_current_length + this->_void_space_from_start));
}


template <typename _Ty>
_Ty* decx::utils::Dynamic_Array<_Ty>::operator[](const size_t _index)
{
    return this->_begin_ptr + _index;
}


template <typename _Ty>
void decx::utils::Dynamic_Array<_Ty>::del(const uint64_t index)
{
    if (this->_current_length > 0) {
        if (index > this->_current_length - 1) {
            return;
        }
        else if (index == this->_current_length - 1) {
            this->pop_back();
        }
        else {
            memcpy(this->_buffer.ptr, this->_begin_ptr + index + 1, (this->_current_length - index - 1) * sizeof(_Ty));
            memcpy(this->_begin_ptr + index, this->_buffer.ptr, (this->_current_length - index - 1) * sizeof(_Ty));
            --this->_current_length;
        }
    }
}


template <typename _Ty>
uint64_t decx::utils::Dynamic_Array<_Ty>::size() const
{
    return this->_current_length;
}


template<typename _Ty>
void decx::utils::Dynamic_Array<_Ty>::move_ahead()
{
    if (this->_current_length != 0) {
        _Ty* _dst = this->_data.ptr;
        _Ty* _src = this->_begin_ptr;

        memcpy(this->_buffer.ptr, this->_begin_ptr, this->_current_length * sizeof(_Ty));
        memcpy(this->_data.ptr, this->_buffer.ptr, this->_current_length * sizeof(_Ty));

        this->_void_space_from_start = 0;
        this->_begin_ptr = this->_data.ptr;
        this->_end_ptr = this->_begin_ptr + this->_current_length + 1;
    }
    else {
        this->_void_space_from_start = 0;
        this->_begin_ptr = this->_data.ptr;
        this->_end_ptr = this->_data.ptr;
    }
}


template<typename _Ty>
void decx::utils::Dynamic_Array<_Ty>::memory_expansion()
{
    this->_memory_capacity += Array_Expansion_Length;
    if (decx::alloc::_host_virtual_page_realloc(&this->_buffer, this->_memory_capacity * sizeof(_Ty))) {
        return;
    }

    memcpy(this->_buffer.ptr, this->_begin_ptr, this->_current_length * sizeof(_Ty));

    if (decx::alloc::_host_virtual_page_realloc(&this->_data, this->_memory_capacity * sizeof(_Ty))) {
        return;
    }

    memcpy(this->_data.ptr, this->_buffer.ptr, this->_current_length * sizeof(_Ty));
    this->_begin_ptr = this->_data.ptr;
    this->_end_ptr = this->_begin_ptr + this->_current_length;
    this->_void_space_from_start = 0;
}


template<typename _Ty>
template<typename... Args>
void decx::utils::Dynamic_Array<_Ty>::emplace_back(Args&&... args)
{
    if (!this->check_vaild_space_req()) {
        if (this->_void_space_from_start > 0) {
            this->move_ahead();
        }
        else {
            this->memory_expansion();
        }
    }
    new (this->_end_ptr) _Ty{ std::forward<Args>(args)... };
    ++this->_end_ptr;
    ++this->_current_length;
}


template<typename _Ty>
_Ty* decx::utils::Dynamic_Array<_Ty>::pop_front()
{
    if (this->_current_length > 1) {
        ++this->_begin_ptr;
        --this->_current_length;
        ++this->_void_space_from_start;
        return (this->_begin_ptr - 1);
    }
    else {
        --this->_current_length;
        ++this->_void_space_from_start;
        return (this->_begin_ptr);
    }
}


template<typename _Ty>
_Ty* decx::utils::Dynamic_Array<_Ty>::pop_back()
{
    --this->_current_length;
    --this->_end_ptr;
    return (this->_end_ptr);
}


template <typename _Ty>
_Ty* decx::utils::Dynamic_Array<_Ty>::front()
{
    return (this->_begin_ptr);
}


template <typename _Ty>
_Ty* decx::utils::Dynamic_Array<_Ty>::back()
{
    return (this->_end_ptr - 1);
}



template<typename _Ty>
void decx::utils::Dynamic_Array<_Ty>::clear()
{
    this->_current_length = 0;
    this->_begin_ptr = this->_data.ptr;
    this->_end_ptr = this->_data.ptr;
    this->_void_space_from_start = 0;
}


template<typename _Ty>
decx::utils::Dynamic_Array<_Ty>::~Dynamic_Array()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_data);
    decx::alloc::_host_virtual_page_dealloc(&this->_buffer);
}


#endif

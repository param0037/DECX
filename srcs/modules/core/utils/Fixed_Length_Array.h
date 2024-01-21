/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _FIXED_LENGTH_ARRAY_H_
#define _FIXED_LENGTH_ARRAY_H_


#include "../basic.h"



namespace decx
{
    namespace utils {
        template <typename _Ty>
        class Fixed_Length_Array;
    }
}


template <typename _Ty>
class decx::utils::Fixed_Length_Array
{
private:
    _Ty* _data;

    size_t _memory_capacity;
    size_t _current_length;

    // The first element
    _Ty* _begin_ptr;
    // After the last element, not accessible !
    _Ty* _end_ptr;


    /**
     * @brief Judge if _end_ptr will exceed the end of the physical memory
     *
     * @return true if will not exceed (safe)
     * @return false if will exceed (Not safe)
     */
    bool check_vaild_space_req();

public:
    Fixed_Length_Array();


    Fixed_Length_Array(const size_t length);


    /**
    * Only called after the default construction function is called !
    * Since after this class is constructed defaultly, the physical capacity is zero
    * @param _new_length : The physical capacity of the array
    */
    void define_capacity(const size_t _new_length);


    /**
     * @brief Get the epscified element of the array by the index
     * @param _index : The index of the element
     * @return _Ty* pointer of the epscified element. If there is no element in the array
     * the return value will be unexpected
     */
    _Ty& operator[](const size_t _index);


    const _Ty* get_const_ptr(const uint64_t _index) const;


    _Ty* get_ptr(const uint64_t _index);


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
     * @brief Get the size of the array, NOT the physical capacity
     *
     * @return The size of the array
     */
    size_t size();


    /**
     * @brief Clear all the element and move Array::_begin_ptr to Array::_data - 1
     *
     */
    void clear();


    uint64_t effective_size() const;



    ~Fixed_Length_Array();
};




template<typename _Ty>
bool decx::utils::Fixed_Length_Array<_Ty>::check_vaild_space_req()
{
    return (this->_memory_capacity > this->_current_length);
}


template <typename _Ty>
decx::utils::Fixed_Length_Array<_Ty>::Fixed_Length_Array()
{
    this->_data = NULL;
    this->_begin_ptr = NULL;
    this->_end_ptr = NULL;
}


template <typename _Ty>
uint64_t decx::utils::Fixed_Length_Array<_Ty>::effective_size() const
{
    return this->_current_length;
}



template <typename _Ty>
decx::utils::Fixed_Length_Array<_Ty>::Fixed_Length_Array(const size_t length)
{
    this->_data = NULL;
    this->_data = (_Ty*)malloc(length * sizeof(_Ty));
    if (this->_data == NULL) {
        Print_Error_Message(4, "Error : Internal error\n");
        exit(-1);
    }
    this->_memory_capacity = length;
    this->_current_length = 0;

    this->_begin_ptr = this->_data - 1;
    this->_end_ptr = this->_data;
}




template <typename _Ty>
void decx::utils::Fixed_Length_Array<_Ty>::define_capacity(const size_t length)
{
    if (this->_data != NULL) {
        free(this->_data);
        this->_data == NULL;
    }
    this->_data = (_Ty*)malloc(length * sizeof(_Ty));

    if (this->_data == NULL) {
        Print_Error_Message(4, "Error : Internal error\n");
        exit(-1);
    }
    this->_memory_capacity = length;
    this->_current_length = 0;

    this->_begin_ptr = this->_data - 1;
    this->_end_ptr = this->_data;
}



template <typename _Ty>
_Ty& decx::utils::Fixed_Length_Array<_Ty>::operator[](const size_t _index)
{
    return *(this->_begin_ptr + _index);
}



template <typename _Ty>
const _Ty* decx::utils::Fixed_Length_Array<_Ty>::get_const_ptr(const uint64_t _index) const
{
    return (this->_begin_ptr + _index);
}


template <typename _Ty>
_Ty* decx::utils::Fixed_Length_Array<_Ty>::get_ptr(const uint64_t _index)
{
    return (this->_begin_ptr + _index);
}



template<typename _Ty>
template<typename... Args>
void decx::utils::Fixed_Length_Array<_Ty>::emplace_back(Args&&... args)
{
    if (!this->check_vaild_space_req()) {
        Print_Error_Message(4, "Error : Internal error\n");
        exit(-1);
    }
    if (this->_current_length == 0) {
        ++this->_begin_ptr;
    }
    new (this->_end_ptr) _Ty{ std::forward<Args>(args)... };
    ++this->_end_ptr;
    ++this->_current_length;
}



template <typename _Ty>
size_t decx::utils::Fixed_Length_Array<_Ty>::size()
{
    return this->_current_length;
}


template <typename _Ty>
_Ty* decx::utils::Fixed_Length_Array<_Ty>::front()
{
    return (this->_begin_ptr);
}


template <typename _Ty>
_Ty* decx::utils::Fixed_Length_Array<_Ty>::back()
{
    return (this->_end_ptr - 1);
}



template<typename _Ty>
_Ty* decx::utils::Fixed_Length_Array<_Ty>::pop_back()
{
    --this->_current_length;
    --this->_end_ptr;
    return (this->_end_ptr);
}


template<typename _Ty>
void decx::utils::Fixed_Length_Array<_Ty>::clear()
{
    this->_current_length = 0;
    this->_begin_ptr = this->_data - 1;
    this->_end_ptr = this->_data;
}


template<typename _Ty>
decx::utils::Fixed_Length_Array<_Ty>::~Fixed_Length_Array()
{
    free(this->_data);
}


#endif
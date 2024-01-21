/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DYNAMIC_ARRAY_H_
#define _DYNAMIC_ARRAY_H_


#include "../basic.h"


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
    _Ty* _data;

    size_t _memory_capacity;
    size_t _current_length;
    size_t _void_space_from_start;

    // The first element
    _Ty* _begin_ptr;
    // After the last element, not accessible !
    _Ty* _end_ptr;

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
    size_t size();


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
    this->_data = NULL;
    this->_void_space_from_start = 0;
    this->_data = (_Ty*)malloc(Array_Initial_Length * sizeof(_Ty));
    if (this->_data == NULL) {
        Print_Error_Message(4, "Error : Internal error\n");
        exit(-1);
    }
    this->_memory_capacity = Array_Initial_Length;
    this->_current_length = 0;

    this->_begin_ptr = this->_data - 1;
    this->_end_ptr = this->_data;
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
size_t decx::utils::Dynamic_Array<_Ty>::size()
{
    return this->_current_length;
}


template<typename _Ty>
void decx::utils::Dynamic_Array<_Ty>::move_ahead()
{
    if (this->_current_length != 0){
        _Ty* _dst = this->_data;
        _Ty* _src = this->_begin_ptr;
        
        size_t i = 0;
        for (i = 0; i < this->_current_length; 
            i += this->_void_space_from_start){
            memcpy(_dst + i, _src + i, this->_void_space_from_start * sizeof(_Ty));
        }
        size_t _L = this->_current_length - (i - this->_void_space_from_start);
        if (_L > 0){
            memcpy(_dst + i - this->_void_space_from_start, _src + i - this->_void_space_from_start, 
            this->_void_space_from_start * sizeof(_Ty));
        }
        this->_void_space_from_start = 0;
        this->_begin_ptr = this->_data;
        this->_end_ptr = this->_begin_ptr + this->_current_length + 1;
    }
    else{
        this->_void_space_from_start = 0;
        this->_begin_ptr = this->_data - 1;
        this->_end_ptr = this->_data;
    }
}


template<typename _Ty>
void decx::utils::Dynamic_Array<_Ty>::memory_expansion()
{
    this->_memory_capacity += Array_Expansion_Length;
    void *new_ptr = malloc(this->_memory_capacity * sizeof(_Ty));
    if (new_ptr == NULL){
        Print_Error_Message(4, "Error : Internal error\n");
        exit(-1);
    }

    memcpy(new_ptr, this->_begin_ptr, this->_current_length * sizeof(_Ty));

    free(this->_data);

    this->_data = static_cast<_Ty*>(new_ptr);
    this->_begin_ptr = this->_data;
    this->_end_ptr = this->_begin_ptr + this->_current_length;
    this->_void_space_from_start = 0;
}


template<typename _Ty>
template<typename... Args>
void decx::utils::Dynamic_Array<_Ty>::emplace_back(Args&&... args)
{
    if (!this->check_vaild_space_req()){
        if (this->_void_space_from_start > 0){
            this->move_ahead();
        }
        else{
            this->memory_expansion();
        }
    }
    if (this->_current_length == 0){
        ++this->_begin_ptr;
    }
    new (this->_end_ptr) _Ty{std::forward<Args>(args)...};
    ++this->_end_ptr;
    ++this->_current_length;
}


template<typename _Ty>
_Ty* decx::utils::Dynamic_Array<_Ty>::pop_front()
{
    if (this->_current_length > 1){
        ++this->_begin_ptr;
        --this->_current_length;
        ++this->_void_space_from_start;
        return (this->_begin_ptr - 1);
    }
    else{
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
    this->_begin_ptr = this->_data - 1;
    this->_end_ptr = this->_data;
    this->_void_space_from_start = 0;
}


template<typename _Ty>
decx::utils::Dynamic_Array<_Ty>::~Dynamic_Array()
{
    free(this->_data);
}



#endif
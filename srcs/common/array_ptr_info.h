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


#ifndef _ARRAY_PTR_INFO_H_
#define _ARRAY_PTR_INFO_H_

#include "../basic.h"
#include "../thread_management/thread_arrange.h"

namespace decx
{
    namespace utils
    {
        template <typename T>
        struct ArrayPtrInfo;


        template <typename T>
        struct ArrayPtrInfo_MT;
    }
}


template <typename T>
struct decx::utils::ArrayPtrInfo
{
    size_t _gap;        // The gap between the pointers (or the size of the space in the medium)
    T* _ptr;            // The forst pointer   
    T** _ptr_arr;       // The pointer array that stores all the pointers
    uint _len;          // The length of the pointer array


    ArrayPtrInfo() {
        this->_ptr = NULL;
        this->_ptr_arr = NULL;
    }


    ArrayPtrInfo(const uint length, const size_t _size, T* __first) {
        this->_len = length;
        this->_gap = _size;
        this->_ptr = __first;

        this->_ptr_arr = (T**)malloc(this->_len * sizeof(T*));
        this->_ptr_arr[0] = this->_ptr;
        for (int i = 1; i < this->_len; ++i) {
            this->_ptr_arr[i] = this->_ptr_arr[i - 1] + this->_gap;
        }
    }

    // void assign

    ~ArrayPtrInfo() {
        if (this->_ptr_arr != NULL)
            free(this->_ptr_arr);
    }
};


template <typename T>
struct decx::utils::ArrayPtrInfo_MT
{
    size_t _thr_proc_size;      // the size of an area process by one thread
    size_t _arr_len;
    decx::utils::ArrayPtrInfo<T>* _apis_arr;

    ArrayPtrInfo_MT() {
        this->_apis_arr = 0;
        this->_arr_len = 0;
        this->_thr_proc_size = NULL;
    }

    /*
    * @param _thr_num : The number of threads
    * @param _first_ptr : The pointer of the first
    * @param page_num : How many areas with the same size in the whole processing area
    * @param _proc_size : The size of an area that processed by one thread
    * @param page_size : The size of a single page
    */
    ArrayPtrInfo_MT(const uint _thr_num, T* _first_ptr, const uint page_num, 
        const size_t _proc_size, const size_t page_size_dst) 
    {
        this->_thr_proc_size = _proc_size;
        this->_arr_len = _thr_num;
        this->_apis_arr = (decx::utils::ArrayPtrInfo<T>*)malloc(this->_arr_len * sizeof(decx::utils::ArrayPtrInfo<T>));
        if (this->_apis_arr == NULL) {
            
            exit(-1);
        }
        new(this->_apis_arr) decx::utils::ArrayPtrInfo<T>(page_num, page_size_dst, _first_ptr);
        for (int i = 1; i < _thr_num; ++i) {
            new(this->_apis_arr + i) decx::utils::ArrayPtrInfo<T>(
                page_num, page_size_dst, _first_ptr + i * this->_thr_proc_size);
        }
    }

    ~ArrayPtrInfo_MT() {
        if (this->_apis_arr != NULL) {
            for (int i = 0; i < this->_arr_len; ++i) {
                this->_apis_arr[i].~ArrayPtrInfo();
            }
            free(this->_apis_arr);
        }
    }
};



#endif
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


#ifndef _DOUBLE_BUFFER_H_
#define _DOUBLE_BUFFER_H_

namespace decx
{
    namespace utils {
        struct BufferStatus_t;
        struct double_buffer_manager;
    }
}


struct decx::utils::BufferStatus_t
{
    /* This is the number of the size of the buffer that this->mem is pointing to,
    in bytes. */
    void* _data = NULL;

    bool
        /* If true, the this buffer is loaded with data most recently. Otherwise, the
        data in it is relatively old. This state can be set by a function called
        decx::utils::set_mutex_memory_state<_Ty1, _Ty2>(MIF*, MIF*) */
        leading,

        /* If true, this buffer is currently being used by calculation units (e.g. CUDA kernels)
        This function is commonly used where device concurrency is needed. Otherwise, this buffer
        is idle. */
        _using;
};


struct decx::utils::double_buffer_manager
{
    // decx::alloc::MIF<void> _MIF1, _MIF2;
    BufferStatus_t _buf1;
    BufferStatus_t _buf2;

    double_buffer_manager() {
        memset(this, 0, sizeof(decx::utils::double_buffer_manager));
    }

    double_buffer_manager(void* _tmp1, void* _tmp2) {
        memset(this, 0, sizeof(decx::utils::double_buffer_manager));
        this->_buf1._data = _tmp1;
        this->_buf2._data = _tmp2;
    }

    void ResetBuf1AsLeading() {
        this->_buf1.leading = true;
        this->_buf2.leading = false;
    }


    void ResetBuf2AsLeading() {
        this->_buf1.leading = false;
        this->_buf2.leading = true;
    }


    void UpdateStatus() {
        this->_buf1.leading = !this->_buf1.leading;
        this->_buf2.leading = !this->_buf2.leading;
    }


    template <typename _ptr_type>
    _ptr_type* GetLeadingBufPtr() {
        return (_ptr_type*)(this->_buf1.leading ? this->_buf1._data : this->_buf2._data);
    }


    template <typename _ptr_type>
    _ptr_type* GetLaggingBufPtr() {
        return (_ptr_type*)(this->_buf2.leading ? this->_buf1._data : this->_buf2._data);
    }


    template <typename _ptr_type>
    _ptr_type* get_buffer1() {
        return (_ptr_type*)this->_buf1._data;
    }


    template <typename _ptr_type>
    _ptr_type* get_buffer2() {
        return (_ptr_type*)this->_buf2._data;
    }
};


#endif
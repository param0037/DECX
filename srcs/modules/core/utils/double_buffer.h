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
    namespace alloc {
        /*
        * This is a struct, which represents a buffer, and the two state words
        * indicates the states.
        */
        template<class T>
        struct MIF
        {
            /* This is the number of the size of the buffer that this->mem is pointing to,
            in bytes. */
            T* mem;

            bool
                /* If true, the this buffer is loaded with data most recently. Otherwise, the
                data in it is relatively old. This state can be set by a function called
                decx::utils::set_mutex_memory_state<_Ty1, _Ty2>(MIF*, MIF*) */
                leading,

                /* If true, this buffer is currently being used by calculation units (e.g. CUDA kernels)
                This function is commonly used where device concurrency is needed. Otherwise, this buffer
                is idle. */
                _using;

            MIF() {
                leading = false;
                _using = false;
                mem = NULL;
            }

            MIF(T* _ptr) {
                leading = false;
                _using = false;
                mem = _ptr;
            }


            MIF(T* _ptr, const bool _leading) {
                leading = _leading;
                _using = false;
                mem = _ptr;
            }
        };
    }


    namespace utils
    {
        template <typename _Ty1, typename _Ty2>
        static inline void set_mutex_memory_state(decx::alloc::MIF<_Ty1>* _set_leading, decx::alloc::MIF<_Ty2>* _set_lagging);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory3_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
            decx::alloc::MIF<_Ty3>* _proc_C);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory3_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
            decx::alloc::MIF<_Ty3>* _proc_C);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory2_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory2_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B);


        template <typename _Ty1, typename _Ty2>
        inline void inverse_mutex_memory_state(decx::alloc::MIF<_Ty1>* _MIF1, decx::alloc::MIF<_Ty2>* _MIF2);
    }
}



template <typename _Ty1, typename _Ty2>
inline void decx::utils::set_mutex_memory_state(decx::alloc::MIF<_Ty1>* _set_leading,
    decx::alloc::MIF<_Ty2>* _set_lagging)
{
    _set_leading->leading = true;
    _set_lagging->leading = false;
}


template <typename _Ty1, typename _Ty2>
inline void decx::utils::inverse_mutex_memory_state(decx::alloc::MIF<_Ty1>* _MIF1, decx::alloc::MIF<_Ty2>* _MIF2)
{
    _MIF1->leading = !_MIF1->leading;
    _MIF2->leading = !_MIF2->leading;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory3_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
    decx::alloc::MIF<_Ty3>* _proc_C)
{
    _proc_A->_using = true;
    _proc_B->_using = true;
    _proc_C->_using = true;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory3_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
    decx::alloc::MIF<_Ty3>* _proc_C)
{
    _proc_A->_using = false;
    _proc_B->_using = false;
    _proc_C->_using = false;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory2_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B)
{
    _proc_A->_using = true;
    _proc_B->_using = true;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory2_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B)
{
    _proc_A->_using = false;
    _proc_B->_using = false;
}



namespace decx
{
    namespace utils {
        struct double_buffer_manager;
    }
}


struct decx::utils::double_buffer_manager
{
    decx::alloc::MIF<void> _MIF1, _MIF2;

    double_buffer_manager(void* _tmp1, void* _tmp2) {
        this->_MIF1.mem = _tmp1;
        this->_MIF2.mem = _tmp2;
    }

    void reset_buffer1_leading() {
        decx::utils::set_mutex_memory_state<void, void>(&this->_MIF1, &this->_MIF2);
    }


    void reset_buffer2_leading() {
        decx::utils::set_mutex_memory_state<void, void>(&this->_MIF2, &this->_MIF1);
    }


    void update_states() {
        decx::utils::inverse_mutex_memory_state<void, void>(&this->_MIF1, &this->_MIF2);
    }

    template <typename _ptr_type>
    _ptr_type* get_leading_ptr() {
        return (_ptr_type*)(this->_MIF1.leading ? this->_MIF1.mem : this->_MIF2.mem);
    }


    template <typename _ptr_type>
    _ptr_type* get_lagging_ptr() {
        return (_ptr_type*)(this->_MIF2.leading ? this->_MIF1.mem : this->_MIF2.mem);
    }


    template <typename _ptr_type>
    _ptr_type* get_buffer1() {
        return (_ptr_type*)this->_MIF1.mem;
    }


    template <typename _ptr_type>
    _ptr_type* get_buffer2() {
        return (_ptr_type*)this->_MIF2.mem;
    }
};


#endif
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _THREAD_ARRANGE_H_
#define _THREAD_ARRANGE_H_

#include "../basic.h"

namespace decx
{
    namespace utils
    {

        typedef struct _thread_arrange_1D
        {
            uint total_thread;
            std::future<void>* _async_thread;

            _thread_arrange_1D(const uint _total_thread, std::future<void>* __async_thread)
            {
                this->total_thread = _total_thread;
                this->_async_thread = __async_thread;
            }

            _thread_arrange_1D(const uint _total_thread)
            {
                this->total_thread = _total_thread;
                this->_async_thread = new std::future<void>[this->total_thread];
            }

            void __sync_all_threads() {
                for (int i = 0; i < this->total_thread; ++i) {
                    this->_async_thread[i].get();
                }
            }

            ~_thread_arrange_1D() {
                delete[] this->_async_thread;
            }
        }_thr_1D;


        typedef struct _thread_arrange_2D
        {
            uint total_thread;
            uint thread_h, thread_w;
            std::future<void>* _async_thread;

            _thread_arrange_2D(const uint _thread_h, const uint _thread_w, std::future<void>* __async_thread)
            {
                this->thread_h = _thread_h;
                this->thread_w = _thread_w;
                this->_async_thread = __async_thread;
                this->total_thread = _thread_h * _thread_w;
            }

            _thread_arrange_2D(const uint _thread_h, const uint _thread_w)
            {
                this->thread_h = _thread_h;
                this->thread_w = _thread_w;
                this->total_thread = _thread_h * _thread_w;
                this->_async_thread = new std::future<void>[this->total_thread];
            }

            void __sync_all_threads() {
                for (int i = 0; i < this->total_thread; ++i) {
                    this->_async_thread[i].get();
                }
            }


            void __sync_all_threads(const uint2 _range) {
                if (_range.y > this->total_thread) {
                    Print_Error_Message(4, "memory out of bound\n");
                    return;
                }
                for (int i = _range.x; i < _range.y; ++i) {
                    this->_async_thread[i].get();
                }
            }


            ~_thread_arrange_2D() {
                delete[] this->_async_thread;
            }
        }_thr_2D;
    }
}



#endif
/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "async_task_executor.h"


decx::async::async_task_executor::async_task_executor(std::condition_variable* _cv_main_thread,
    std::mutex* _mtx_main_thread)
{
    this->_cv_main_thread = _cv_main_thread;
    this->_mtx_main_thread = _mtx_main_thread;

    this->_task_num = 0;
    this->_shutdown = false;
    this->_all_done = true;
    this->_ID = 0;
    this->_idle = true;
}



void decx::async::async_task_executor::_thread_main_loop()
{
    while (!this->_shutdown) {
        std::unique_lock<std::mutex> lock(this->_mtx);
        while (this->_all_done) {
            this->_cv.wait(lock);
        }
        for (int i = 0; i < this->_task_num; ++i) {
            Task* _tk = this->_task_queue[i];
            (*_tk)();
        }
        this->_all_done = true;
        this->_task_num = 0;
        this->_task_queue.clear();

        this->_cv_main_thread->notify_one();
    }
}


void decx::async::async_task_executor::start()
{
    if (this->_exec != NULL) {
        new (this->_exec) std::thread(&decx::async::async_task_executor::_thread_main_loop, this);
    }
}


void decx::async::async_task_executor::bind_thread_addr(std::thread* _addr) 
{
    this->_exec = _addr;
}


void decx::async::async_task_executor::stream_shutdown()
{
    this->_mtx_main_thread->lock();
    this->_shutdown = true;
    this->_mtx_main_thread->unlock();

    if (this->_exec->joinable()) {
        this->_exec->join();
    }
}
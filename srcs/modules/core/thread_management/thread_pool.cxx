/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "thread_pool.h"

void decx::ThreadPool::_find_task_queue_id(size_t* id)
{
    size_t task_que_len = this->current_thread_num;
    size_t res_id = 0,
        least_len = (this->_task_schd)->_task_num;

    if (least_len != 0) {
        for (size_t i = 1; i < task_que_len; ++i)
        {
            decx::ThreadTaskQueue* tmp_iter = this->_task_schd + i;

            size_t current_len = tmp_iter->_task_num;

            if (current_len != 0) {
                if (current_len < least_len)
                    least_len = current_len;
            }
            else {
                least_len = i;
                break;
            }
        }
        *id = least_len;
    }
    else {
        *id = res_id;
    }
}



void decx::ThreadPool::_thread_main_loop(const size_t pool_id)
{
    decx::ThreadTaskQueue* thread_unit = &(this->_task_schd[pool_id]);

    while (!thread_unit->_shutdown)
    {
        std::unique_lock<std::mutex> lock{ thread_unit->_mtx };
        while ((thread_unit->_task_num == 0) && (!thread_unit->_shutdown)) {
            thread_unit->_cv.wait(lock);
        }

        if (thread_unit->_task_num != 0) {
            Task* task = thread_unit->begin_ptr + thread_unit->_task_num - 1;
            (*task)();     // execute the tast
            thread_unit->pop_back();
        }
    }
    return;
}



void decx::ThreadPool::Start()
{
    this->_all_shutdown = false;

    for (int i = 0; i < this->current_thread_num; ++i) {
        new(this->_task_schd + i) decx::ThreadTaskQueue();
    }
    for (size_t i = 0; i < this->current_thread_num; ++i) {
        new(this->_thr_list + i) std::thread(&decx::ThreadPool::_thread_main_loop, this, i);
    }
}



decx::ThreadPool::ThreadPool(const int thread_num, const bool start_at_begin)
{
    this->_all_shutdown = true;
    this->_max_thr_num = MAX_THREAD_NUM;
    this->current_thread_num = thread_num;

    this->_hardware_concurrent = std::thread::hardware_concurrency();

    this->_task_schd = (decx::ThreadTaskQueue*)malloc(this->_max_thr_num * sizeof(decx::ThreadTaskQueue));
    this->_thr_list = (std::thread*)malloc(this->_max_thr_num * sizeof(std::thread));

    this->_sync_label = 0;
    this->_internal_sync_enable = false;

    if (start_at_begin) {
        Start();
    }
}





void decx::ThreadPool::add_thread(const int add_thread_num)
{
    if (this->current_thread_num + add_thread_num > this->_max_thr_num) {
        return;
    }
    else {
        for (int i = 0; i < add_thread_num; ++i) {
            new(this->_task_schd + this->current_thread_num + i) decx::ThreadTaskQueue();
        }
        for (size_t i = 0; i < add_thread_num; ++i) {
            new(this->_thr_list + this->current_thread_num + i) std::thread(
                &decx::ThreadPool::_thread_main_loop, this, i);
        }
    }
}




void decx::ThreadPool::TerminateAllThreads()
{
    for (int i = 0; i < this->current_thread_num; ++i) {
        std::thread* _iter = this->_thr_list + i;
        decx::ThreadTaskQueue* Tschd_iter = this->_task_schd + i;
        {
            std::unique_lock<std::mutex> lck(Tschd_iter->_mtx);
            Tschd_iter->_shutdown = true;
        }
        Tschd_iter->_cv.notify_one();
        _iter->join();
    }

    this->_all_shutdown = true;
}



decx::ThreadPool::~ThreadPool() {
    if (!this->_all_shutdown) {
        TerminateAllThreads();
    }
    for (int i = 0; i < this->current_thread_num; ++i) {
        std::thread* _iter = this->_thr_list + i;
        decx::ThreadTaskQueue* Tschd_iter = this->_task_schd + i;
        Tschd_iter->~ThreadTaskQueue();
        _iter->~thread();
    }

    free(this->_task_schd);
    free(this->_thr_list);
}

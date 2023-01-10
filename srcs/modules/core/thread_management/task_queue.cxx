/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "task_queue.h"

void decx::ThreadTaskQueue::move_ahead()
{
    int idle_size = (int)(this->begin_ptr - this->_task_queue);

    memcpy(this->_task_queue, this->begin_ptr, idle_size * sizeof(Task));
    if (this->_task_num - idle_size > 0)
        memcpy(this->_task_queue + idle_size, this->begin_ptr + idle_size, (this->_task_num - idle_size) * sizeof(Task));
}



decx::ThreadTaskQueue::ThreadTaskQueue() {
    this->_task_queue = (Task*)malloc(64 * sizeof(Task));
    this->begin_ptr = this->_task_queue;
    this->end_ptr = this->_task_queue + 64;

    this->_shutdown = false;
    this->_task_num = 0;
}



void decx::ThreadTaskQueue::pop_back() {
    --this->_task_num;
}



void decx::ThreadTaskQueue::pop_front()
{
    if (this->_task_num > 1) {
        if (this->begin_ptr + 1 == this->end_ptr) {
            this->move_ahead();
            this->begin_ptr = this->_task_queue;
        }
        else {
            ++this->begin_ptr;
        }
    }
    else {
        this->begin_ptr = this->_task_queue;
    }
    --this->_task_num;
}



decx::ThreadTaskQueue::~ThreadTaskQueue() {
    free(this->_task_queue);
}

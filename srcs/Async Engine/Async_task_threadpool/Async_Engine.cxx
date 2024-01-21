/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Async_Engine.h"


#define Async_Engine_Initial_Stream_Number 12
#define Async_Engine_Expand_Stream_Number 1024


decx::async::Async_Engine::Async_Engine()
{
    this->_is_init = false;
}


void decx::async::Async_Engine::Init()
{
    if (!this->_is_init){
        // alloc space for stream_list
        this->_thread_list = (std::thread*)malloc(Async_Engine_Initial_Stream_Number * sizeof(std::thread));
        // alloc space for task_scheduler
        this->_task_schd = (decx::async::_ATE*)malloc(Async_Engine_Initial_Stream_Number * sizeof(decx::async::_ATE));

        this->current_stream_num = 0;
        this->actual_capacity = Async_Engine_Initial_Stream_Number;
        this->_is_init = true;
        printf("Async Engine malloced\n");
    }
}



bool decx::async::Async_Engine::find_available_stream(int* _ID)
{
    for (int i = 0; i < this->current_stream_num; ++i) {
        if (this->_task_schd[i]._idle) {
            *_ID = i;
            this->_task_schd[i]._idle = false;
            return true;
        }
    }
    *_ID = -1;
    return false;
}



void decx::async::Async_Engine::resize()
{
    std::thread* new_addr_threads =
        (std::thread*)malloc((this->actual_capacity + Async_Engine_Expand_Stream_Number) * sizeof(std::thread));
    if (new_addr_threads == NULL) {
        Print_Error_Message(4, "Error : Internal error(s)\n");
        exit(-1);
    }

    decx::async::_ATE* new_addr_ATE =
        (decx::async::_ATE*)malloc((this->actual_capacity + Async_Engine_Expand_Stream_Number) * sizeof(decx::async::_ATE));

    if (new_addr_ATE == NULL) {
        Print_Error_Message(4, "Error : Internal error(s)\n");
        exit(-1);
    }

    memcpy(new_addr_threads, this->_task_schd, this->actual_capacity * sizeof(std::thread));
    memcpy(new_addr_ATE, this->_task_schd, this->actual_capacity * sizeof(decx::async::_ATE));

    free(this->_thread_list);
    this->_thread_list = new_addr_threads;

    free(this->_task_schd);
    this->_task_schd = new_addr_ATE;

    this->actual_capacity += Async_Engine_Expand_Stream_Number;
}



uint32_t decx::async::Async_Engine::add_stream()
{
    int id;
    if (this->find_available_stream(&id)) {     // if found
        return id;
    }
    else {      
        // if not found, new a decx::async_task_executor on memory physically
        // And each extension expands memory space for <Async_Engine_Expand_Stream_Number>
        if (this->current_stream_num + 1 > this->actual_capacity) {
            this->resize();
        }
        new (this->_task_schd + this->current_stream_num) decx::async::async_task_executor(&this->_cv_main_thread,
            &this->_mtx_main_thread);

        this->_task_schd[this->current_stream_num]._idle = false;
        this->_task_schd[this->current_stream_num].bind_thread_addr(this->_thread_list + this->current_stream_num);

        this->_task_schd[this->current_stream_num].start();

        uint _res_id = this->current_stream_num;
        this->_task_schd[this->current_stream_num]._ID = this->current_stream_num;
        ++this->current_stream_num;
        return _res_id;
    }
}


void decx::async::Async_Engine::shutdown()
{
    for (int i = 0; i < this->current_stream_num; ++i) {
        this->_task_schd[i].stream_shutdown();
    }
}


uint decx::async::Async_Engine::get_current_stream_num()
{
    return this->current_stream_num;
}



decx::async::_ATE* decx::async::Async_Engine::operator[](const size_t dex)
{
    return (this->_task_schd + dex);
}



decx::async::Async_Engine::~Async_Engine()
{
    this->shutdown();

    free(this->_task_schd);
    free(this->_thread_list);
}


_DECX_API_ decx::async::_ATE* decx::async::_get_ATE_by_id(const uint64_t id)
{
    return &decx::async::async_engine._task_schd[id];
}


_DECX_API_ uint64_t decx::async::AsyncEngine_add_stream()
{
    return decx::async::async_engine.add_stream();
}